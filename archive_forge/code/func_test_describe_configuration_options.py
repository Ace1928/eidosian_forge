import random
import time
from functools import partial
from tests.compat import unittest
from boto.beanstalk.wrapper import Layer1Wrapper
import boto.beanstalk.response as response
def test_describe_configuration_options(self):
    result = self.beanstalk.describe_configuration_options()
    self.assertIsInstance(result, response.DescribeConfigurationOptionsResponse)