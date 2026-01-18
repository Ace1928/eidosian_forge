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
def test_delete_rule_list(self):
    schema = {'far': properties.Schema(properties.Schema.LIST, schema=properties.Schema(properties.Schema.MAP, schema={'red': properties.Schema(properties.Schema.STRING), 'check': properties.Schema(properties.Schema.STRING)}))}
    data = {'far': [{'red': 'blue'}, {'red': 'roses'}]}
    props = properties.Properties(schema, data)
    rule = translation.TranslationRule(props, translation.TranslationRule.DELETE, ['far', 'red'])
    tran = translation.Translation(props)
    tran.set_rules([rule])
    self.assertTrue(tran.has_translation('far.red'))
    self.assertIsNone(tran.translate('far.red'))
    self.assertIsNone(tran.resolved_translations['far.red'])