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
def do_import_info(gc, args):
    """Print import methods available from Glance."""
    try:
        import_info = gc.images.get_import_info()
    except exc.HTTPNotFound:
        utils.exit('Target Glance does not support Image Import workflow')
    else:
        utils.print_dict(import_info)