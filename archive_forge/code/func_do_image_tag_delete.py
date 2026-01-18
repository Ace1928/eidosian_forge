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
@utils.arg('image_id', metavar='<IMAGE_ID>', help=_('ID of the image from which to delete tag.'))
@utils.arg('tag_value', metavar='<TAG_VALUE>', help=_('Value of the tag.'))
def do_image_tag_delete(gc, args):
    """Delete the tag associated with the given image."""
    if not (args.image_id and args.tag_value):
        utils.exit('Unable to delete tag. Specify image_id and tag_value')
    else:
        gc.image_tags.delete(args.image_id, args.tag_value)