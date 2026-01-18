import random
import time
from functools import partial
from tests.compat import unittest
from boto.beanstalk.wrapper import Layer1Wrapper
import boto.beanstalk.response as response
def test_15_list_available_solution_stacks(self):
    result = self.beanstalk.list_available_solution_stacks()
    self.assertIsInstance(result, response.ListAvailableSolutionStacksResponse)
    self.assertIn('32bit Amazon Linux running Tomcat 6', result.solution_stacks)