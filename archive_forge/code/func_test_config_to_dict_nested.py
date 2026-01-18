import os
import tempfile
import unittest
from webtest import TestApp
import pecan
from pecan.tests import PecanTestCase
def test_config_to_dict_nested(self):
    from pecan import configuration
    'have more than one level nesting and convert to dict'
    conf = configuration.initconf()
    nested = {'one': {'two': 2}}
    conf['nested'] = nested
    to_dict = conf.to_dict()
    assert isinstance(to_dict, dict)
    assert to_dict['server']['host'] == '0.0.0.0'
    assert to_dict['server']['port'] == '8080'
    assert to_dict['app']['modules'] == []
    assert to_dict['app']['root'] is None
    assert to_dict['app']['static_root'] == 'public'
    assert to_dict['app']['template_path'] == ''
    assert to_dict['nested']['one']['two'] == 2