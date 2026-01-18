import copy
from unittest import mock
from heat.common import exception
from heat.engine.cfn import functions as cfn_funcs
from heat.engine import function
from heat.engine.hot import functions as hot_funcs
from heat.engine import parameters
from heat.engine import properties
from heat.engine import translation
from heat.tests import common
def test_add_rule_list_with_custom_value_path(self):
    schema = {'far': properties.Schema(properties.Schema.LIST, schema=properties.Schema(properties.Schema.MAP, schema={'red': properties.Schema(properties.Schema.LIST, schema=properties.Schema(properties.Schema.STRING)), 'blue': properties.Schema(properties.Schema.MAP)}))}
    data = {'far': [{'blue': {'black': {'white': 'daisy', 'check': ['one']}}}, {'red': ['roses']}]}
    props = properties.Properties(schema, data)
    rule = translation.TranslationRule(props, translation.TranslationRule.ADD, ['far', 'red'], value_name='blue', custom_value_path=['black', 'check'])
    tran = translation.Translation(props)
    tran.set_rules([rule])
    self.assertTrue(tran.has_translation('far.0.red'))
    result = tran.translate('far.0.red', data['far'][0].get('red'), data['far'][0])
    self.assertEqual(['one'], result)
    self.assertEqual(['one'], tran.resolved_translations['far.0.red'])
    self.assertEqual(['roses'], tran.translate('far.1.red', data['far'][1]['red'], data['far'][1]))