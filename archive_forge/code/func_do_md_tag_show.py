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
@utils.arg('namespace', metavar='<NAMESPACE>', help=_('Name of the namespace to which the tag belongs.'))
@utils.arg('tag', metavar='<TAG>', help=_('Name of the tag.'))
def do_md_tag_show(gc, args):
    """Describe a specific metadata definitions tag inside a namespace."""
    tag = gc.metadefs_tag.get(args.namespace, args.tag)
    _tag_show(tag)