import collections
from oslo_serialization import jsonutils
from heat.common import exception
from heat.common.i18n import _
from heat.common import param_utils
from heat.engine import constraints as constr
from heat.engine import function
from heat.engine.hot import parameters as hot_param
from heat.engine import parameters
from heat.engine import support
from heat.engine import translation as trans
@classmethod
def schema_to_parameters_and_properties(cls, schema, template_type='cfn'):
    """Convert a schema to template parameters and properties.

        This can be used to generate a provider template that matches the
        given properties schemata.

        :param schema: A resource type's properties_schema
        :returns: A tuple of params and properties dicts

        ex: input:  {'foo': {'Type': 'List'}}
            output: {'foo': {'Type': 'CommaDelimitedList'}},
                    {'foo': {'Fn::Split': {'Ref': 'foo'}}}

        ex: input:  {'foo': {'Type': 'String'}, 'bar': {'Type': 'Map'}}
            output: {'foo': {'Type': 'String'}, 'bar': {'Type': 'Json'}},
                    {'foo': {'Ref': 'foo'}, 'bar': {'Ref': 'bar'}}
        """

    def param_prop_def_items(name, schema, template_type):
        if template_type == 'hot':
            param_def = cls._hot_param_def_from_prop(schema)
            prop_def = cls._hot_prop_def_from_prop(name, schema)
        else:
            param_def = cls._param_def_from_prop(schema)
            prop_def = cls._prop_def_from_prop(name, schema)
        return ((name, param_def), (name, prop_def))
    if not schema:
        return ({}, {})
    param_prop_defs = [param_prop_def_items(n, s, template_type) for n, s in schemata(schema).items() if s.implemented]
    param_items, prop_items = zip(*param_prop_defs)
    return (dict(param_items), dict(prop_items))