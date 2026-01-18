import functools
from unittest import mock
import uuid
from keystoneauth1 import loading
from keystoneauth1.loading import base
from keystoneauth1 import plugin
from keystoneauth1.tests.unit import utils
def mock_plugin(loader=MockLoader):

    def _wrapper(f):

        @functools.wraps(f)
        def inner(*args, **kwargs):
            with mock.patch.object(base, 'get_plugin_loader') as m:
                m.return_value = loader()
                args = list(args) + [m]
                return f(*args, **kwargs)
        return inner
    return _wrapper