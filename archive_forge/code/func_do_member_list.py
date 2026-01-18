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
@utils.arg('--image-id', metavar='<IMAGE_ID>', required=True, help=_('Image to display members of.'))
def do_member_list(gc, args):
    """Describe sharing permissions by image."""
    members = gc.image_members.list(args.image_id)
    columns = ['Image ID', 'Member ID', 'Status']
    utils.print_list(members, columns)