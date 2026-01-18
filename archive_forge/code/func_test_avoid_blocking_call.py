from unittest import mock
import ddt
from os_win import _utils
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
@mock.patch('eventlet.tpool.execute')
@mock.patch('eventlet.getcurrent')
@ddt.data(mock.Mock(), None)
def test_avoid_blocking_call(self, gt_parent, mock_get_current_gt, mock_execute):
    mock_get_current_gt.return_value.parent = gt_parent
    mock_execute.return_value = mock.sentinel.ret_val

    def fake_blocking_func(*args, **kwargs):
        self.assertEqual((mock.sentinel.arg,), args)
        self.assertEqual(dict(kwarg=mock.sentinel.kwarg), kwargs)
        return mock.sentinel.ret_val
    fake_blocking_func_decorated = _utils.avoid_blocking_call_decorator(fake_blocking_func)
    ret_val = fake_blocking_func_decorated(mock.sentinel.arg, kwarg=mock.sentinel.kwarg)
    self.assertEqual(mock.sentinel.ret_val, ret_val)
    if gt_parent:
        mock_execute.assert_called_once_with(fake_blocking_func, mock.sentinel.arg, kwarg=mock.sentinel.kwarg)
    else:
        self.assertFalse(mock_execute.called)