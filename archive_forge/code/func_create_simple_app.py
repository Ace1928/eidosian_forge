import sys
import time
from unittest import mock
import uuid
import warnings
import fixtures
from oslo_log import log as logging
import oslotest.base as oslotest
import requests
import webob
import webtest
def create_simple_app(self, *args, **kwargs):
    return webtest.TestApp(self.create_simple_middleware(*args, **kwargs))