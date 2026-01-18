from unittest import mock
from oslo_serialization import jsonutils
from heat.common import exception
from heat.engine import constraints
from heat.engine import function
from heat.engine.hot import functions as hot_funcs
from heat.engine.hot import parameters as hot_param
from heat.engine import parameters
from heat.engine import plugin_manager
from heat.engine import properties
from heat.engine import resources
from heat.engine import rsrc_defn
from heat.engine import support
from heat.engine import translation
from heat.tests import common
def test_no_mismatch_in_update_policy(self):
    manager = plugin_manager.PluginManager('heat.engine.resources')
    resource_mapping = plugin_manager.PluginMapping('resource')
    res_plugin_mappings = resource_mapping.load_all(manager)
    all_resources = {}
    for mapping in res_plugin_mappings:
        name, cls = mapping
        all_resources[name] = cls

    def check_update_policy(resource_type, prop_key, prop, update=False):
        if prop.update_allowed:
            update = True
        sub_schema = prop.schema
        if sub_schema:
            for sub_prop_key, sub_prop in sub_schema.items():
                if not update:
                    self.assertEqual(update, sub_prop.update_allowed, "Mismatch in update policies: resource %(res)s, properties '%(prop)s' and '%(nested_prop)s'." % {'res': resource_type, 'prop': prop_key, 'nested_prop': sub_prop_key})
                if sub_prop_key == '*':
                    check_update_policy(resource_type, prop_key, sub_prop, update)
                else:
                    check_update_policy(resource_type, sub_prop_key, sub_prop, update)
    for resource_type, resource_class in all_resources.items():
        props_schemata = properties.schemata(resource_class.properties_schema)
        for prop_key, prop in props_schemata.items():
            check_update_policy(resource_type, prop_key, prop)