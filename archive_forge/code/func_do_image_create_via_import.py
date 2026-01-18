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
@utils.schema_args(get_image_schema, omit=['locations', 'os_hidden'])
@utils.arg('--hidden', type=strutils.bool_from_string, metavar='[True|False]', default=None, dest='os_hidden', help='If true, image will not appear in default image list response.')
@utils.arg('--property', metavar='<key=value>', action='append', default=[], help=_('Arbitrary property to associate with image. May be used multiple times.'))
@utils.arg('--file', metavar='<FILE>', help=_('Local file that contains disk image to be uploaded during creation. Alternatively, the image data can be passed to the client via stdin.'))
@utils.arg('--progress', action='store_true', default=False, help=_('Show upload progress bar.'))
@utils.arg('--import-method', metavar='<METHOD>', default=utils.env('OS_IMAGE_IMPORT_METHOD', default=None), help=_("Import method used for Image Import workflow. Valid values can be retrieved with import-info command. Defaults to env[OS_IMAGE_IMPORT_METHOD] or if that is undefined uses 'glance-direct' if data is provided using --file or stdin. Otherwise, simply creates an image record if no import-method and no data is supplied"))
@utils.arg('--uri', metavar='<IMAGE_URL>', default=None, help=_('URI to download the external image.'))
@utils.arg('--remote-region', metavar='<GLANCE_REGION>', default=None, help=_('REMOTE_GLANCE_REGION to download the image.'))
@utils.arg('--remote-image-id', metavar='<REMOTE_IMAGE_ID>', default=None, help=_('The IMAGE ID of the image of remote glance, which needsto be imported with glance-download'))
@utils.arg('--remote-service-interface', metavar='<REMOTE_SERVICE_INTERFACE>', default='public', help=_('The Remote Glance Service Interface for glance-download'))
@utils.arg('--store', metavar='<STORE>', default=utils.env('OS_IMAGE_STORE', default=None), help='Backend store to upload image to.')
@utils.arg('--stores', metavar='<STORES>', default=utils.env('OS_IMAGE_STORES', default=None), help=_('Stores to upload image to if multi-stores import available. Comma separated list. Available stores can be listed with "stores-info" call.'))
@utils.arg('--all-stores', type=strutils.bool_from_string, metavar='[True|False]', default=None, dest='os_all_stores', help=_('"all-stores" can be ued instead of "stores"-list to indicate that image should be imported into all available stores.'))
@utils.arg('--allow-failure', type=strutils.bool_from_string, metavar='[True|False]', dest='os_allow_failure', default=utils.env('OS_IMAGE_ALLOW_FAILURE', default=True), help=_('Indicator if all stores listed (or available) must succeed. "True" by default meaning that we allow some stores to fail and the status can be monitored from the image metadata. If this is set to "False" the import will be reverted should any of the uploads fail. Only usable with "stores" or "all-stores".'))
@utils.on_data_require_fields(DATA_FIELDS)
def do_image_create_via_import(gc, args):
    """EXPERIMENTAL: Create a new image via image import.

    Use the interoperable image import workflow to create an image.  This
    command is designed to be backward compatible with the current image-create
    command, so its behavior is as follows:

    * If an import-method is specified (either on the command line or through
      the OS_IMAGE_IMPORT_METHOD environment variable, then you must provide a
      data source appropriate to that method (for example, --file for
      glance-direct, or --uri for web-download).
    * If no import-method is specified AND you provide either a --file or
      data to stdin, the command will assume you are using the 'glance-direct'
      import-method and will act accordingly.
    * If no import-method is specified and no data is supplied via --file or
      stdin, the command will simply create an image record in 'queued' status.
    """
    schema = gc.schemas.get('image')
    _args = [(x[0].replace('-', '_'), x[1]) for x in vars(args).items()]
    fields = dict(filter(lambda x: x[1] is not None and (x[0] == 'property' or schema.is_core_property(x[0])), _args))
    raw_properties = fields.pop('property', [])
    for datum in raw_properties:
        key, value = datum.split('=', 1)
        fields[key] = value
    file_name = fields.pop('file', None)
    using_stdin = hasattr(sys.stdin, 'isatty') and (not sys.stdin.isatty())
    if args.import_method is None and (file_name or using_stdin):
        args.import_method = 'glance-direct'
    if args.import_method == 'copy-image':
        utils.exit("Import method 'copy-image' cannot be used while creating the image.")
    import_methods = gc.images.get_import_info().get('import-methods')
    if args.import_method and args.import_method not in import_methods.get('value'):
        utils.exit("Import method '%s' is not valid for this cloud. Valid values can be retrieved with import-info command." % args.import_method)
    backend = None
    stores = getattr(args, 'stores', None)
    all_stores = getattr(args, 'os_all_stores', None)
    if args.store and (stores or all_stores) or (stores and all_stores):
        utils.exit('Only one of --store, --stores and --all-stores can be provided')
    elif args.store:
        backend = args.store
        _validate_backend(backend, gc)
    elif stores:
        fields.pop('stores')
        stores = str(stores).split(',')
        for store in stores:
            _validate_backend(store, gc)
    if args.import_method is None:
        if args.uri:
            utils.exit('You cannot use --uri without specifying an import method.')
    if args.import_method == 'glance-direct':
        if backend and (not (file_name or using_stdin)):
            utils.exit('--store option should only be provided with --file option or stdin for the glance-direct import method.')
        if stores and (not (file_name or using_stdin)):
            utils.exit('--stores option should only be provided with --file option or stdin for the glance-direct import method.')
        if all_stores and (not (file_name or using_stdin)):
            utils.exit('--all-stores option should only be provided with --file option or stdin for the glance-direct import method.')
        if args.uri:
            utils.exit('You cannot specify a --uri with the glance-direct import method.')
        if file_name is not None and os.access(file_name, os.R_OK) is False:
            utils.exit('File %s does not exist or user does not have read privileges to it.' % file_name)
        if file_name is not None and using_stdin:
            utils.exit('You cannot use both --file and stdin with the glance-direct import method.')
        if not file_name and (not using_stdin):
            utils.exit('You must specify a --file or provide data via stdin for the glance-direct import method.')
    if args.import_method == 'web-download':
        if backend and (not args.uri):
            utils.exit('--store option should only be provided with --uri option for the web-download import method.')
        if stores and (not args.uri):
            utils.exit('--stores option should only be provided with --uri option for the web-download import method.')
        if all_stores and (not args.uri):
            utils.exit('--all-stores option should only be provided with --uri option for the web-download import method.')
        if not args.uri:
            utils.exit("URI is required for web-download import method. Please use '--uri <uri>'.")
        if file_name:
            utils.exit('You cannot specify a --file with the web-download import method.')
        if using_stdin:
            utils.exit('You cannot pass data via stdin with the web-download import method.')
    if args.import_method == 'glance-download':
        if not (args.remote_region and args.remote_image_id):
            utils.exit('REMOTE GlANCE REGION and REMOTE IMAGE ID are required for glance-download import method. Please use --remote-region <region> and --remote-image-id <remote-image-id>.')
        if args.uri:
            utils.exit('You cannot specify a --uri with the glance-download import method.')
        if file_name:
            utils.exit('You cannot specify a --file with the glance-download import method.')
        if using_stdin:
            utils.exit('You cannot pass data via stdin with the glance-download import method.')
    image = gc.images.create(**fields)
    try:
        args.id = image['id']
        if args.import_method:
            if utils.get_data_file(args) is not None:
                args.size = None
                do_image_stage(gc, args)
            args.from_create = True
            args.stores = stores
            do_image_import(gc, args)
        image = gc.images.get(args.id)
    finally:
        utils.print_image(image)