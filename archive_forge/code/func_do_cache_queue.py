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
@utils.arg('id', metavar='<IMAGE_ID>', nargs='+', help=_('ID of image(s) to queue for caching.'))
def do_cache_queue(gc, args):
    """Queue image(s) for caching."""
    if not gc.endpoint_provided:
        utils.exit('Direct server endpoint needs to be provided. Do not use loadbalanced or catalog endpoints.')
    for args_id in args.id:
        try:
            gc.cache.queue(args_id)
        except exc.HTTPForbidden:
            msg = _("You are not permitted to queue the image '%s' for caching." % args_id)
            utils.print_err(msg)
        except exc.HTTPException as e:
            msg = _("'%s': Unable to queue image '%s' for caching." % (e, args_id))
            utils.print_err(msg)