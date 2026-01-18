import contextlib
import os
from unittest import mock
import testtools
from troveclient.apiclient import exceptions
from troveclient import base
from troveclient import common
from troveclient import utils
def get_mock_resource_obj(self):
    base.Resource.__init__ = mock.Mock(return_value=None)
    robj = base.Resource()
    robj._loaded = False
    return robj