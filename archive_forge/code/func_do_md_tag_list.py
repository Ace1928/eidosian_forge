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
@utils.arg('namespace', metavar='<NAMESPACE>', help=_('Name of namespace.'))
def do_md_tag_list(gc, args):
    """List metadata definitions tags inside a specific namespace."""
    tags = gc.metadefs_tag.list(args.namespace)
    columns = ['name']
    column_settings = {'description': {'max_width': 50, 'align': 'l'}}
    utils.print_list(tags, columns, field_settings=column_settings)