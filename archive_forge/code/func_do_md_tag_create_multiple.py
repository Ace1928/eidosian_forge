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
@utils.arg('namespace', metavar='<NAMESPACE>', help=_('Name of the namespace the tags will belong to.'))
@utils.arg('--names', metavar='<NAMES>', required=True, help=_('A comma separated list of tag names.'))
@utils.arg('--delim', metavar='<DELIM>', required=False, help=_('The delimiter used to separate the names (if none is provided then the default is a comma).'))
@utils.arg('--append', default=False, action='store_true', required=False, help=_('Append the new tags to the existing ones instead ofoverwriting them'))
def do_md_tag_create_multiple(gc, args):
    """Create new metadata definitions tags inside a namespace."""
    delim = args.delim or ','
    tags = []
    names_list = args.names.split(delim)
    for name in names_list:
        name = name.strip()
        if name:
            tags.append(name)
    if not tags:
        utils.exit('Please supply at least one tag name. For example: --names Tag1')
    fields = {'tags': tags, 'append': args.append}
    new_tags = gc.metadefs_tag.create_multiple(args.namespace, **fields)
    columns = ['name']
    column_settings = {'description': {'max_width': 50, 'align': 'l'}}
    utils.print_list(new_tags, columns, field_settings=column_settings)