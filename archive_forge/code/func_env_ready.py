import random
import time
from functools import partial
from tests.compat import unittest
from boto.beanstalk.wrapper import Layer1Wrapper
import boto.beanstalk.response as response
@classmethod
def env_ready(cls, env_name, desired_status):
    result = cls.beanstalk.describe_environments(application_name=cls.app_name, environment_names=env_name)
    status = result.environments[0].status
    return status == desired_status