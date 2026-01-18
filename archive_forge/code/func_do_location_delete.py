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
@utils.arg('--url', metavar='<URL>', action='append', required=True, help=_('URL of location to remove. May be used multiple times.'))
@utils.arg('id', metavar='<IMAGE_ID>', help=_('ID of image whose locations are to be removed.'))
def do_location_delete(gc, args):
    """Remove locations (and related metadata) from an image."""
    gc.images.delete_locations(args.id, set(args.url))