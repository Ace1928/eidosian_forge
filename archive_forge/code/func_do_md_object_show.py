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
@utils.arg('namespace', metavar='<NAMESPACE>', help=_('Name of namespace the object belongs.'))
@utils.arg('object', metavar='<OBJECT>', help=_('Name of an object.'))
@utils.arg('--max-column-width', metavar='<integer>', default=80, help=_('The max column width of the printed table.'))
def do_md_object_show(gc, args):
    """Describe a specific metadata definitions object inside a namespace."""
    obj = gc.metadefs_object.get(args.namespace, args.object)
    _object_show(obj, int(args.max_column_width))