import boto
from boto.configservice.exceptions import NoSuchConfigurationRecorderException
from tests.compat import unittest
def test_describe_configuration_recorders(self):
    response = self.configservice.describe_configuration_recorders()
    self.assertIn('ConfigurationRecorders', response)