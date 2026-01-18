from unittest import mock
from oslotest import base as test_base
from oslo_log import helpers
def test_log_decorator_for_static(self):
    """Test that LOG.debug is called with proper arguments."""

    @helpers.log_method_call
    def _static_method():
        pass

    class test_class(object):

        @staticmethod
        @helpers.log_method_call
        def test_staticmethod(arg1, arg2, arg3, *args, **kwargs):
            pass
    data = {'caller': 'static', 'method_name': '_static_method', 'args': (), 'kwargs': {}}
    with mock.patch('logging.Logger.debug') as debug:
        _static_method()
        debug.assert_called_with(mock.ANY, data)
    args = tuple(range(6))
    kwargs = {'kwarg1': 6, 'kwarg2': 7}
    data = {'caller': 'static', 'method_name': 'test_staticmethod', 'args': args, 'kwargs': kwargs}
    with mock.patch('logging.Logger.debug') as debug:
        test_class.test_staticmethod(*args, **kwargs)
        debug.assert_called_with(mock.ANY, data)