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
@utils.arg('--detail', default=False, action='store_true', help='Shows details of stores. Admin only.')
def do_stores_info(gc, args):
    """Print available backends from Glance."""
    try:
        if args.detail:
            stores_info = gc.images.get_stores_info_detail()
        else:
            stores_info = gc.images.get_stores_info()
    except exc.HTTPNotFound:
        utils.exit('Multi Backend support is not enabled')
    else:
        utils.print_dict(stores_info)