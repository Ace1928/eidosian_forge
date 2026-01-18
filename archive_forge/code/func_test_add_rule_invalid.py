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
def test_add_rule_invalid(self):
    schema = {'far': properties.Schema(properties.Schema.MAP, schema={'red': properties.Schema(properties.Schema.STRING)}), 'bar': properties.Schema(properties.Schema.STRING)}
    data = {'far': 'tran', 'bar': 'dak'}
    props = properties.Properties(schema, data)
    rule = translation.TranslationRule(props, translation.TranslationRule.ADD, ['far'], [props.get('bar')])
    tran = translation.Translation(props)
    tran.set_rules([rule])
    self.assertTrue(tran.has_translation('far'))
    ex = self.assertRaises(ValueError, tran.translate, 'far', 'tran')
    self.assertEqual('Incorrect translation rule using - cannot resolve Add rule for non-list translation value "far".', str(ex))