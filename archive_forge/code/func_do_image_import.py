import json
import os
import sys
from oslo_utils import strutils
from glanceclient._i18n import _
from glanceclient.common import progressbar
from glanceclient.common import utils
from glanceclient import exc
from glanceclient.v2 import cache
from glanceclient.v2 import image_members
from glanceclient.v2 import image_schema
from glanceclient.v2 import images
from glanceclient.v2 import namespace_schema
from glanceclient.v2 import resource_type_schema
from glanceclient.v2 import tasks
@utils.arg('--import-method', metavar='<METHOD>', default='glance-direct', help=_('Import method used for Image Import workflow. Valid values can be retrieved with import-info command and the default "glance-direct" is used with "image-stage".'))
@utils.arg('--uri', metavar='<IMAGE_URL>', default=None, help=_('URI to download the external image.'))
@utils.arg('--remote-region', metavar='<REMOTE_GLANCE_REGION>', default=None, help=_('REMOTE GLANCE REGION to download the image.'))
@utils.arg('--remote-image-id', metavar='<REMOTE_IMAGE_ID>', default=None, help=_('The IMAGE ID of the image of remote glance, which needsto be imported with glance-download'))
@utils.arg('--remote-service-interface', metavar='<REMOTE_SERVICE_INTERFACE>', default='public', help=_('The Remote Glance Service Interface for glance-download'))
@utils.arg('id', metavar='<IMAGE_ID>', help=_('ID of image to import.'))
@utils.arg('--store', metavar='<STORE>', default=utils.env('OS_IMAGE_STORE', default=None), help='Backend store to upload image to.')
@utils.arg('--stores', metavar='<STORES>', default=utils.env('OS_IMAGE_STORES', default=None), help='Stores to upload image to if multi-stores import available.')
@utils.arg('--all-stores', type=strutils.bool_from_string, metavar='[True|False]', default=None, dest='os_all_stores', help=_('"all-stores" can be ued instead of "stores"-list to indicate that image should be imported all available stores.'))
@utils.arg('--allow-failure', type=strutils.bool_from_string, metavar='[True|False]', dest='os_allow_failure', default=utils.env('OS_IMAGE_ALLOW_FAILURE', default=True), help=_('Indicator if all stores listed (or available) must succeed. "True" by default meaning that we allow some stores to fail and the status can be monitored from the image metadata. If this is set to "False" the import will be reverted should any of the uploads fail. Only usable with "stores" or "all-stores".'))
def do_image_import(gc, args):
    """Initiate the image import taskflow."""
    backend = getattr(args, 'store', None)
    stores = getattr(args, 'stores', None)
    all_stores = getattr(args, 'os_all_stores', None)
    allow_failure = getattr(args, 'os_allow_failure', True)
    uri = getattr(args, 'uri', None)
    remote_region = getattr(args, 'remote_region', None)
    remote_image_id = getattr(args, 'remote_image_id', None)
    remote_service_interface = getattr(args, 'remote_service_interface', None)
    if not getattr(args, 'from_create', False):
        if args.store and (stores or all_stores) or (stores and all_stores):
            utils.exit('Only one of --store, --stores and --all-stores can be provided')
        elif args.store:
            backend = args.store
            _validate_backend(backend, gc)
        elif stores:
            stores = str(stores).split(',')
        if stores:
            for store in stores:
                _validate_backend(store, gc)
    if getattr(args, 'from_create', False):
        gc.images.image_import(args.id, args.import_method, args.uri, remote_region=remote_region, remote_image_id=remote_image_id, remote_service_interface=remote_service_interface, backend=backend, stores=stores, all_stores=all_stores, allow_failure=allow_failure)
        return
    try:
        import_methods = gc.images.get_import_info().get('import-methods')
    except exc.HTTPNotFound:
        utils.exit('Target Glance does not support Image Import workflow')
    if args.import_method not in import_methods.get('value'):
        utils.exit("Import method '%s' is not valid for this cloud. Valid values can be retrieved with import-info command." % args.import_method)
    if args.import_method == 'web-download' and (not args.uri):
        utils.exit('Provide URI for web-download import method.')
    if args.uri and args.import_method != 'web-download':
        utils.exit("Import method should be 'web-download' if URI is provided.")
    if args.import_method == 'glance-download' and (not (remote_region and remote_image_id)):
        utils.exit("Provide REMOTE_IMAGE_ID and remote-region for 'glance-download' import method.")
    if remote_region and args.import_method != 'glance-download':
        utils.exit("Import method should be 'glance-download' if REMOTE REGION is provided.")
    if remote_image_id and args.import_method != 'glance-download':
        utils.exit("Import method should be 'glance-download' if REMOTE IMAGE ID is provided.")
    if args.import_method == 'copy-image' and (not (stores or all_stores)):
        utils.exit("Provide either --stores or --all-stores for 'copy-image' import method.")
    image = gc.images.get(args.id)
    container_format = image.get('container_format')
    disk_format = image.get('disk_format')
    if not (container_format and disk_format):
        utils.exit("The 'container_format' and 'disk_format' properties must be set on an image before it can be imported.")
    image_status = image.get('status')
    if args.import_method == 'glance-direct':
        if image_status != 'uploading':
            utils.exit("The 'glance-direct' import method can only be applied to an image in status 'uploading'")
    if args.import_method == 'web-download':
        if image_status != 'queued':
            utils.exit("The 'web-download' import method can only be applied to an image in status 'queued'")
    if args.import_method == 'copy-image':
        if image_status != 'active':
            utils.exit("The 'copy-image' import method can only be used on an image with status 'active'.")
    gc.images.image_import(args.id, args.import_method, uri=uri, remote_region=remote_region, remote_image_id=remote_image_id, remote_service_interface=remote_service_interface, backend=backend, stores=stores, all_stores=all_stores, allow_failure=allow_failure)
    image = gc.images.get(args.id)
    utils.print_image(image)