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
@utils.arg('--allow-md5-fallback', action='store_true', default=utils.env('OS_IMAGE_ALLOW_MD5_FALLBACK', default=False), help=_('If os_hash_algo and os_hash_value properties are available on the image, they will be used to validate the downloaded image data. If the indicated secure hash algorithm is not available on the client, the download will fail. Use this flag to indicate that in such a case the legacy MD5 image checksum should be used to validate the downloaded data. You can also set the environment variable OS_IMAGE_ALLOW_MD5_FALLBACK to any value to activate this option.'))
@utils.arg('--file', metavar='<FILE>', help=_('Local file to save downloaded image data to. If this is not specified and there is no redirection the image data will not be saved.'))
@utils.arg('id', metavar='<IMAGE_ID>', help=_('ID of image to download.'))
@utils.arg('--progress', action='store_true', default=False, help=_('Show download progress bar.'))
def do_image_download(gc, args):
    """Download a specific image."""
    if sys.stdout.isatty() and args.file is None:
        msg = 'No redirection or local file specified for downloaded image data. Please specify a local file with --file to save downloaded image or redirect output to another source.'
        utils.exit(msg)
    try:
        body = gc.images.data(args.id, allow_md5_fallback=args.allow_md5_fallback)
    except (exc.HTTPForbidden, exc.HTTPException) as e:
        msg = "Unable to download image '%s'. (%s)" % (args.id, e)
        utils.exit(msg)
    if body.wrapped is None:
        msg = 'Image %s has no data.' % args.id
        utils.exit(msg)
    if args.progress:
        body = progressbar.VerboseIteratorWrapper(body, len(body))
    utils.save_image(body, args.file)