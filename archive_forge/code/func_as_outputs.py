import collections
import functools
from oslo_utils import strutils
from heat.common.i18n import _
from heat.engine import constraints as constr
from heat.engine import support
from oslo_log import log as logging
@staticmethod
def as_outputs(resource_name, resource_class, template_type='cfn'):
    """Dict of Output entries for a provider template with resource name.

        :param resource_name: logical name of the resource
        :param resource_class: resource implementation class
        :returns: The attributes of the specified resource_class as a template
                  Output map
        """
    attr_schema = {}
    for name, schema_data in resource_class.attributes_schema.items():
        schema = Schema.from_attribute(schema_data)
        if schema.support_status.status != support.HIDDEN:
            attr_schema[name] = schema
    attr_schema.update(resource_class.base_attributes_schema)
    attribs = Attributes._make_attributes(attr_schema).items()
    outp = dict(((n, att.as_output(resource_name, template_type)) for n, att in attribs))
    outp['OS::stack_id'] = _stack_id_output(resource_name, template_type)
    return outp