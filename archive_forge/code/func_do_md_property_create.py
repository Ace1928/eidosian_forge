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
@utils.arg('namespace', metavar='<NAMESPACE>', help=_('Name of namespace the property will belong.'))
@utils.arg('--name', metavar='<NAME>', required=True, help=_('Internal name of a property.'))
@utils.arg('--title', metavar='<TITLE>', required=True, help=_('Property name displayed to the user.'))
@utils.arg('--schema', metavar='<SCHEMA>', required=True, help=_('Valid JSON schema of a property.'))
@utils.arg('--type', metavar='<TYPE>', required=True, help=_('Type of the property'))
def do_md_property_create(gc, args):
    """Create a new metadata definitions property inside a namespace."""
    try:
        schema = json.loads(args.schema)
    except ValueError:
        utils.exit('Schema is not a valid JSON object.')
    else:
        fields = {'name': args.name, 'title': args.title, 'type': args.type}
        fields.update(schema)
        new_property = gc.metadefs_property.create(args.namespace, **fields)
        utils.print_dict(new_property)