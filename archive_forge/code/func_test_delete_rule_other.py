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
def test_delete_rule_other(self):
    schema = {'far': properties.Schema(properties.Schema.STRING)}
    data = {'far': 'one'}
    props = properties.Properties(schema, data)
    rule = translation.TranslationRule(props, translation.TranslationRule.DELETE, ['far'])
    tran = translation.Translation(props)
    tran.set_rules([rule])
    self.assertTrue(tran.has_translation('far'))
    self.assertIsNone(tran.translate('far'))
    self.assertIsNone(tran.resolved_translations['far'])