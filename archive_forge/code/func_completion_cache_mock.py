import contextlib
import os
from unittest import mock
import testtools
from troveclient.apiclient import exceptions
from troveclient import base
from troveclient import common
from troveclient import utils
@contextlib.contextmanager
def completion_cache_mock(*arg, **kwargs):
    yield