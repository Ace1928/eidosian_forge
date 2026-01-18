import os
import tempfile
import unittest
from webtest import TestApp
import pecan
from pecan.tests import PecanTestCase
def test_config_to_dict(self):
    from pecan import configuration
    conf = configuration.initconf()
    assert isinstance(conf, configuration.Config)
    to_dict = conf.to_dict()
    assert isinstance(to_dict, dict)
    assert to_dict['server']['host'] == '0.0.0.0'
    assert to_dict['server']['port'] == '8080'
    assert to_dict['app']['modules'] == []
    assert to_dict['app']['root'] is None
    assert to_dict['app']['static_root'] == 'public'
    assert to_dict['app']['template_path'] == ''