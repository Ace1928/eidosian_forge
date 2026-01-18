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
@utils.arg('--name', metavar='<NAME>', default=None, help=_('New name of an object.'))
@utils.arg('--schema', metavar='<SCHEMA>', default=None, help=_('Valid JSON schema of an object.'))
def do_md_object_update(gc, args):
    """Update metadata definitions object inside a namespace."""
    fields = {}
    if args.name:
        fields['name'] = args.name
    if args.schema:
        try:
            schema = json.loads(args.schema)
        except ValueError:
            utils.exit('Schema is not a valid JSON object.')
        else:
            fields.update(schema)
    new_object = gc.metadefs_object.update(args.namespace, args.object, **fields)
    _object_show(new_object)