import collections
import contextlib
import datetime as dt
import itertools
import pydoc
import re
import tenacity
import weakref
from oslo_config import cfg
from oslo_log import log as logging
from oslo_utils import excutils
from oslo_utils import reflection
from heat.common import exception
from heat.common.i18n import _
from heat.common import identifier
from heat.common import short_id
from heat.common import timeutils
from heat.engine import attributes
from heat.engine.cfn import template as cfn_tmpl
from heat.engine import clients
from heat.engine.clients import default_client_plugin
from heat.engine import environment
from heat.engine import event
from heat.engine import function
from heat.engine.hot import template as hot_tmpl
from heat.engine import node_data
from heat.engine import properties
from heat.engine import resources
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.engine import status
from heat.engine import support
from heat.engine import sync_point
from heat.engine import template
from heat.objects import resource as resource_objects
from heat.objects import resource_data as resource_data_objects
from heat.objects import resource_properties_data as rpd_objects
from heat.rpc import client as rpc_client
@classmethod
def resource_to_template(cls, resource_type, template_type='cfn'):
    """Generate a provider template that mirrors the resource.

        :param resource_type: The resource type to be displayed in the template
        :param template_type: the template type to generate, cfn or hot.
        :returns: A template where the resource's properties_schema is mapped
            as parameters, and the resource's attributes_schema is mapped as
            outputs
        """
    props_schema = {}
    for name, schema_dict in cls.properties_schema.items():
        schema = properties.Schema.from_legacy(schema_dict)
        if schema.support_status.status != support.HIDDEN:
            props_schema[name] = schema
    params, props = properties.Properties.schema_to_parameters_and_properties(props_schema, template_type)
    resource_name = cls.__name__
    outputs = attributes.Attributes.as_outputs(resource_name, cls, template_type)
    description = 'Initial template of %s' % resource_name
    return cls.build_template_dict(resource_name, resource_type, template_type, params, props, outputs, description)