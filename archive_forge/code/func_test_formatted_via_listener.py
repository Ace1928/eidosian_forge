from taskflow import engines
from taskflow import formatters
from taskflow.listeners import logging as logging_listener
from taskflow.patterns import linear_flow
from taskflow import states
from taskflow import test
from taskflow.test import mock
from taskflow.test import utils as test_utils
@mock.patch('taskflow.formatters.FailureFormatter._format_node')
def test_formatted_via_listener(self, mock_format_node):
    mock_format_node.return_value = 'A node'
    flo = self._make_test_flow()
    e = engines.load(flo)
    with logging_listener.DynamicLoggingListener(e):
        self.assertRaises(RuntimeError, e.run)
    self.assertTrue(mock_format_node.called)