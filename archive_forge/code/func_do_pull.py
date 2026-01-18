from zunclient.common import cliutils as utils
from zunclient.common import utils as zun_utils
@utils.arg('image', metavar='<image>', help='Name of the image')
@utils.arg('host', metavar='<host>', help='Name or UUID of the host')
def do_pull(cs, args):
    """Pull an image into a host."""
    opts = {}
    opts['repo'] = args.image
    opts['host'] = args.host
    _show_image(cs.images.create(**opts))