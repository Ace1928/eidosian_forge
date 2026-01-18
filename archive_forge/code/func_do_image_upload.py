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
@utils.arg('--file', metavar='<FILE>', help=_('Local file that contains disk image to be uploaded. Alternatively, images can be passed to the client via stdin.'))
@utils.arg('--size', metavar='<IMAGE_SIZE>', type=int, help=_('Size in bytes of image to be uploaded. Default is to get size from provided data object but this is supported in case where size cannot be inferred.'), default=None)
@utils.arg('--progress', action='store_true', default=False, help=_('Show upload progress bar.'))
@utils.arg('id', metavar='<IMAGE_ID>', help=_('ID of image to upload data to.'))
@utils.arg('--store', metavar='<STORE>', default=utils.env('OS_IMAGE_STORE', default=None), help='Backend store to upload image to.')
def do_image_upload(gc, args):
    """Upload data for a specific image."""
    backend = None
    if args.store:
        backend = args.store
        _validate_backend(backend, gc)
    image_data = utils.get_data_file(args)
    if args.progress:
        filesize = utils.get_file_size(image_data)
        if filesize is not None:
            image_data = progressbar.VerboseFileWrapper(image_data, filesize)
    gc.images.upload(args.id, image_data, args.size, backend=backend)