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
@utils.arg('image_id', metavar='<IMAGE_ID>', help=_('Image from which to remove member.'))
@utils.arg('member_id', metavar='<MEMBER_ID>', help=_('Tenant to remove as member.'))
def do_member_delete(gc, args):
    """Delete image member."""
    if not (args.image_id and args.member_id):
        utils.exit('Unable to delete member. Specify image_id and member_id')
    else:
        gc.image_members.delete(args.image_id, args.member_id)