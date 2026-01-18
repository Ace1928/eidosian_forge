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
@utils.arg('namespace', metavar='<NAMESPACE>', help=_('Name of namespace to describe.'))
@utils.arg('--resource-type', metavar='<RESOURCE_TYPE>', help=_('Applies prefix of given resource type associated to a namespace to all properties of a namespace.'), default=None)
@utils.arg('--max-column-width', metavar='<integer>', default=80, help=_('The max column width of the printed table.'))
def do_md_namespace_show(gc, args):
    """Describe a specific metadata definitions namespace.

    Lists also the namespace properties, objects and resource type
    associations.
    """
    kwargs = {}
    if args.resource_type:
        kwargs['resource_type'] = args.resource_type
    namespace = gc.metadefs_namespace.get(args.namespace, **kwargs)
    _namespace_show(namespace, int(args.max_column_width))