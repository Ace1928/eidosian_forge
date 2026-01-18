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
@utils.schema_args(get_resource_type_schema)
def do_md_resource_type_associate(gc, args):
    """Associate resource type with a metadata definitions namespace."""
    schema = gc.schemas.get('metadefs/resource_type')
    _args = [(x[0].replace('-', '_'), x[1]) for x in vars(args).items()]
    fields = dict(filter(lambda x: x[1] is not None and schema.is_core_property(x[0]), _args))
    resource_type = gc.metadefs_resource_type.associate(args.namespace, **fields)
    utils.print_dict(resource_type)