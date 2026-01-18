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
@utils.arg('property', metavar='<PROPERTY>', help=_('Name of a property.'))
@utils.arg('--max-column-width', metavar='<integer>', default=80, help=_('The max column width of the printed table.'))
def do_md_object_property_show(gc, args):
    """Describe a specific metadata definitions property inside an object."""
    obj = gc.metadefs_object.get(args.namespace, args.object)
    try:
        prop = obj['properties'][args.property]
        prop['name'] = args.property
    except KeyError:
        utils.exit('Property %s not found in object %s.' % (args.property, args.object))
    utils.print_dict(prop, int(args.max_column_width))