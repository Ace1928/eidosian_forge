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
@utils.arg('image_id', metavar='<IMAGE_ID>', help=_('Image to be updated with the given tag.'))
@utils.arg('tag_value', metavar='<TAG_VALUE>', help=_('Value of the tag.'))
def do_image_tag_update(gc, args):
    """Update an image with the given tag."""
    if not (args.image_id and args.tag_value):
        utils.exit('Unable to update tag. Specify image_id and tag_value')
    else:
        gc.image_tags.update(args.image_id, args.tag_value)
        image = gc.images.get(args.image_id)
        image = [image]
        columns = ['ID', 'Tags']
        utils.print_list(image, columns)