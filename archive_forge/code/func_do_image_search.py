from zunclient.common import cliutils as utils
from zunclient.common import utils as zun_utils
@utils.arg('image', metavar='<image>', help='Name of the image')
@utils.arg('--image_driver', metavar='<image-driver>', choices=['glance', 'docker'], default='docker', help='Name of the image driver (glance, docker)')
@utils.arg('--exact-match', default=False, action='store_true', help='exact match image name')
def do_image_search(cs, args):
    """Print list of available images from repository based on user query."""
    opts = {}
    opts['image_driver'] = args.image_driver
    opts['exact_match'] = args.exact_match
    images = cs.images.search_image(args.image, **opts)
    columns = ('ID', 'Name', 'Tags', 'Status', 'Size', 'Metadata')
    utils.print_list(images, columns, {'versions': zun_utils.print_list_field('versions')}, sortby_index=None)