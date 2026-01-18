import random
import time
from functools import partial
from tests.compat import unittest
from boto.beanstalk.wrapper import Layer1Wrapper
import boto.beanstalk.response as response
def test_describe_applications(self):
    result = self.beanstalk.describe_applications()
    self.assertIsInstance(result, response.DescribeApplicationsResponse)