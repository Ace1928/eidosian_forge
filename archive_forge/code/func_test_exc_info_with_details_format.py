from taskflow import engines
from taskflow import formatters
from taskflow.listeners import logging as logging_listener
from taskflow.patterns import linear_flow
from taskflow import states
from taskflow import test
from taskflow.test import mock
from taskflow.test import utils as test_utils
@mock.patch('taskflow.formatters.FailureFormatter._format_node')
def test_exc_info_with_details_format(self, mock_format_node):
    mock_format_node.return_value = 'A node'
    flo = self._make_test_flow()
    e = engines.load(flo)
    self.assertRaises(RuntimeError, e.run)
    fails = e.storage.get_execute_failures()
    self.assertEqual(1, len(fails))
    self.assertIn('Broken', fails)
    fail = fails['Broken']
    e.storage.set_atom_intention('Broken', states.EXECUTE)
    f = formatters.FailureFormatter(e)
    exc_info, details = f.format(fail, self._broken_atom_matcher)
    self.assertEqual(3, len(exc_info))
    self.assertTrue(mock_format_node.called)