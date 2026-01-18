import string
from taskflow import exceptions as exc
from taskflow import test
def test_no_looping(self):
    causes = []
    for a in string.ascii_lowercase:
        try:
            cause = causes[-1]
        except IndexError:
            cause = None
        causes.append(exc.TaskFlowException('%s broken' % a, cause=cause))
    e = causes[0]
    last_e = causes[-1]
    e._cause = last_e
    self.assertIsNotNone(e.pformat())