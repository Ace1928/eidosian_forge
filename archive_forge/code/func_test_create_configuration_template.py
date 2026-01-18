import random
import time
from functools import partial
from tests.compat import unittest
from boto.beanstalk.wrapper import Layer1Wrapper
import boto.beanstalk.response as response
def test_create_configuration_template(self):
    self.create_application()
    result = self.beanstalk.create_configuration_template(application_name=self.app_name, template_name=self.template, solution_stack_name='32bit Amazon Linux running Tomcat 6')
    self.assertIsInstance(result, response.CreateConfigurationTemplateResponse)
    self.assertEqual(result.solution_stack_name, '32bit Amazon Linux running Tomcat 6')