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
@utils.arg('namespace', metavar='<NAMESPACE>', help=_('Name of the namespace.'))
@utils.schema_args(get_namespace_schema, omit=['namespace', 'property_count', 'properties', 'tag_count', 'tags', 'object_count', 'objects', 'resource_types'])
def do_md_namespace_create(gc, args):
    """Create a new metadata definitions namespace."""
    schema = gc.schemas.get('metadefs/namespace')
    _args = [(x[0].replace('-', '_'), x[1]) for x in vars(args).items()]
    fields = dict(filter(lambda x: x[1] is not None and schema.is_core_property(x[0]), _args))
    namespace = gc.metadefs_namespace.create(**fields)
    _namespace_show(namespace)