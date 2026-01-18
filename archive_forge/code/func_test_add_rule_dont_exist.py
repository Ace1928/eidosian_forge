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
def test_add_rule_dont_exist(self):
    schema = {'far': properties.Schema(properties.Schema.LIST, schema=properties.Schema(properties.Schema.MAP, schema={'red': properties.Schema(properties.Schema.STRING)})), 'bar': properties.Schema(properties.Schema.STRING)}
    data = {'bar': 'dak'}
    props = properties.Properties(schema, copy.copy(data))
    rule = translation.TranslationRule(props, translation.TranslationRule.ADD, ['far'], [{'red': props.get('bar')}])
    tran = translation.Translation(props)
    tran.set_rules([rule])
    self.assertTrue(tran.has_translation('far'))
    result = tran.translate('far')
    self.assertEqual([{'red': 'dak'}], result)
    self.assertEqual([{'red': 'dak'}], tran.resolved_translations['far'])