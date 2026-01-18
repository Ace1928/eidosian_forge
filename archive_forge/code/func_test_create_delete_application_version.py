import random
import time
from functools import partial
from tests.compat import unittest
from boto.beanstalk.wrapper import Layer1Wrapper
import boto.beanstalk.response as response
def test_create_delete_application_version(self):
    app_result = self.beanstalk.create_application(application_name=self.app_name)
    self.assertIsInstance(app_result, response.CreateApplicationResponse)
    self.assertEqual(app_result.application.application_name, self.app_name)
    version_result = self.beanstalk.create_application_version(application_name=self.app_name, version_label=self.app_version)
    self.assertIsInstance(version_result, response.CreateApplicationVersionResponse)
    self.assertEqual(version_result.application_version.version_label, self.app_version)
    result = self.beanstalk.delete_application_version(application_name=self.app_name, version_label=self.app_version)
    self.assertIsInstance(result, response.DeleteApplicationVersionResponse)
    result = self.beanstalk.delete_application(application_name=self.app_name)
    self.assertIsInstance(result, response.DeleteApplicationResponse)