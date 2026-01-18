import random
import time
from functools import partial
from tests.compat import unittest
from boto.beanstalk.wrapper import Layer1Wrapper
import boto.beanstalk.response as response
def test_describe_configuration_settings(self):
    result = self.beanstalk.describe_configuration_settings(application_name=self.app_name, environment_name=self.environment)
    self.assertIsInstance(result, response.DescribeConfigurationSettingsResponse)