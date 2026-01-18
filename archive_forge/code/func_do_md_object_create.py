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
@utils.arg('namespace', metavar='<NAMESPACE>', help=_('Name of namespace the object will belong.'))
@utils.arg('--name', metavar='<NAME>', required=True, help=_('Internal name of an object.'))
@utils.arg('--schema', metavar='<SCHEMA>', required=True, help=_('Valid JSON schema of an object.'))
def do_md_object_create(gc, args):
    """Create a new metadata definitions object inside a namespace."""
    try:
        schema = json.loads(args.schema)
    except ValueError:
        utils.exit('Schema is not a valid JSON object.')
    else:
        fields = {'name': args.name}
        fields.update(schema)
        new_object = gc.metadefs_object.create(args.namespace, **fields)
        _object_show(new_object)