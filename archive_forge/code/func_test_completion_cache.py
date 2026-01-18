import contextlib
import os
from unittest import mock
import testtools
from troveclient.apiclient import exceptions
from troveclient import base
from troveclient import common
from troveclient import utils
def test_completion_cache(self):
    manager = base.Manager()
    mode = 'w'
    cache_type = 'unittest'
    obj_class = mock.Mock
    with manager.completion_cache(cache_type, obj_class, mode):
        pass
    os.makedirs = mock.Mock(side_effect=OSError)
    with manager.completion_cache(cache_type, obj_class, mode):
        pass