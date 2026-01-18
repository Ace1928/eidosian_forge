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
@utils.arg('namespace', metavar='<NAMESPACE>', help=_('Name of the namespace the tag will belong to.'))
@utils.arg('--name', metavar='<NAME>', required=True, help=_('The name of the new tag to add.'))
def do_md_tag_create(gc, args):
    """Add a new metadata definitions tag inside a namespace."""
    name = args.name.strip()
    if name:
        new_tag = gc.metadefs_tag.create(args.namespace, name)
        _tag_show(new_tag)
    else:
        utils.exit('Please supply at least one non-blank tag name.')