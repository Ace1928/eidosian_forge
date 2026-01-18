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
@utils.arg('--resource-types', metavar='<RESOURCE_TYPES>', action='append', help=_('Resource type to filter namespaces.'))
@utils.arg('--visibility', metavar='<VISIBILITY>', help=_('Visibility parameter to filter namespaces.'))
@utils.arg('--page-size', metavar='<SIZE>', default=None, type=int, help=_('Number of namespaces to request in each paginated request.'))
def do_md_namespace_list(gc, args):
    """List metadata definitions namespaces."""
    filter_keys = ['resource_types', 'visibility']
    filter_items = [(key, getattr(args, key, None)) for key in filter_keys]
    filters = dict([item for item in filter_items if item[1] is not None])
    kwargs = {'filters': filters}
    if args.page_size is not None:
        kwargs['page_size'] = args.page_size
    namespaces = gc.metadefs_namespace.list(**kwargs)
    columns = ['namespace']
    utils.print_list(namespaces, columns)