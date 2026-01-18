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
@utils.arg('id', metavar='<IMAGE_ID>', help=_('ID of image to update.'))
@utils.arg('--store', metavar='<STORE_ID>', required=True, help=_('Store to delete image from.'))
def do_stores_delete(gc, args):
    """Delete image from specific store."""
    try:
        gc.images.delete_from_store(args.store, args.id)
    except exc.HTTPNotFound:
        utils.exit('Multi Backend support is not enabled or Image/store not found.')
    except (exc.HTTPForbidden, exc.HTTPException) as e:
        msg = "Unable to delete image '%s' from store '%s'. (%s)" % (args.id, args.store, e)
        utils.exit(msg)