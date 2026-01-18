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
@utils.arg('--store', metavar='<STORE>', default=utils.env('OS_IMAGE_STORE', default=None), help='Backend store to upload image to.')
@utils.on_data_require_fields(DATA_FIELDS)
def do_image_create(gc, args):
    """Create a new image."""
    schema = gc.schemas.get('image')
    _args = [(x[0].replace('-', '_'), x[1]) for x in vars(args).items()]
    fields = dict(filter(lambda x: x[1] is not None and (x[0] == 'property' or schema.is_core_property(x[0])), _args))
    raw_properties = fields.pop('property', [])
    for datum in raw_properties:
        key, value = datum.split('=', 1)
        fields[key] = value
    backend = args.store
    file_name = fields.pop('file', None)
    using_stdin = hasattr(sys.stdin, 'isatty') and (not sys.stdin.isatty())
    if args.store and (not (file_name or using_stdin)):
        utils.exit('--store option should only be provided with --file option or stdin.')
    if backend:
        _validate_backend(backend, gc)
    if file_name is not None and os.access(file_name, os.R_OK) is False:
        utils.exit('File %s does not exist or user does not have read privileges to it' % file_name)
    image = gc.images.create(**fields)
    try:
        if utils.get_data_file(args) is not None:
            backend = fields.get('store', None)
            args.id = image['id']
            args.size = None
            do_image_upload(gc, args)
            image = gc.images.get(args.id)
    finally:
        utils.print_image(image)