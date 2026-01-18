import string
from taskflow import exceptions as exc
from taskflow import test
def test_cause_pformat(self):
    capture = None
    try:
        raise exc.TaskFlowException('broken', cause=IOError('dead'))
    except Exception as e:
        capture = e
    self.assertIsNotNone(capture)
    self.assertGreater(0, len(capture.pformat()))