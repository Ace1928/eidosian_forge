import collections
import functools
from taskflow import states
from taskflow import test
from taskflow.types import notifier as nt
def test_bad_notify(self):

    def call_me(state, details):
        pass
    notifier = nt.Notifier()
    self.assertRaises(KeyError, notifier.register, nt.Notifier.ANY, call_me, kwargs={'details': 5})