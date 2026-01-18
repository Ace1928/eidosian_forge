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
@utils.arg('id', metavar='<IMAGE_ID>', help=_('ID of image to get tasks.'))
def do_image_tasks(gc, args):
    """Get tasks associated with image"""
    columns = ['Message', 'Status', 'Updated at']
    if args.verbose:
        columns_to_prepend = ['Image Id', 'Task Id']
        columns_to_extend = ['User Id', 'Request Id', 'Result', 'Owner', 'Input', 'Expires at']
        columns = columns_to_prepend + columns + columns_to_extend
    try:
        tasks = gc.images.get_associated_image_tasks(args.id)
        utils.print_dict_list(tasks['tasks'], columns)
    except exc.HTTPNotFound:
        utils.exit('Image %s not found.' % args.id)
    except exc.HTTPNotImplemented:
        utils.exit('Server does not support image tasks API (v2.12)')