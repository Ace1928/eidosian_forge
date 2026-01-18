import boto
from boto.configservice.exceptions import NoSuchConfigurationRecorderException
from tests.compat import unittest
def test_handle_no_such_configuration_recorder(self):
    with self.assertRaises(NoSuchConfigurationRecorderException):
        self.configservice.describe_configuration_recorders(configuration_recorder_names=['non-existant-recorder'])