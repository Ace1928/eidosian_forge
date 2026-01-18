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
def test_invalid_translation_rule(self):
    props = properties.Properties({}, {})
    exc = self.assertRaises(ValueError, translation.TranslationRule, props, 'EatTheCookie', mock.ANY, mock.ANY)
    self.assertEqual('There is no rule EatTheCookie. List of allowed rules is: Add, Replace, Delete, Resolve.', str(exc))
    exc = self.assertRaises(ValueError, translation.TranslationRule, props, translation.TranslationRule.ADD, 'networks.network', 'value')
    self.assertEqual('"translation_path" should be non-empty list with path to translate.', str(exc))
    exc = self.assertRaises(ValueError, translation.TranslationRule, props, translation.TranslationRule.ADD, [], mock.ANY)
    self.assertEqual('"translation_path" should be non-empty list with path to translate.', str(exc))
    exc = self.assertRaises(ValueError, translation.TranslationRule, props, translation.TranslationRule.ADD, ['any'], 'value', 'value_name', 'some_path')
    self.assertEqual('"value_path", "value" and "value_name" are mutually exclusive and cannot be specified at the same time.', str(exc))
    exc = self.assertRaises(ValueError, translation.TranslationRule, props, translation.TranslationRule.ADD, ['any'], 'value')
    self.assertEqual('"value" must be list type when rule is Add.', str(exc))