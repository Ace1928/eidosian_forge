import string
from taskflow import exceptions as exc
from taskflow import test
def test_invalid_pformat_indent(self):
    ex = exc.TaskFlowException('Broken')
    self.assertRaises(ValueError, ex.pformat, indent=-100)