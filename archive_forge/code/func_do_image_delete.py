from zunclient.common import cliutils as utils
from zunclient.common import utils as zun_utils
@utils.arg('id', metavar='<uuid>', help='UUID of image to delete')
def do_image_delete(cs, args):
    """Delete a specified image."""
    opts = {}
    opts['image_id'] = args.id
    cs.images.delete(**opts)