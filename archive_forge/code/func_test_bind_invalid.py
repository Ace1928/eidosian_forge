import collections
import functools
import threading
import time
from taskflow import test
from taskflow.utils import threading_utils as tu
def test_bind_invalid(self):
    self.assertRaises(ValueError, self.bundle.bind, 1)
    for k in ['after_start', 'before_start', 'before_join', 'after_join']:
        kwargs = {k: 1}
        self.assertRaises(ValueError, self.bundle.bind, lambda: tu.daemon_thread(_spinner, self.death), **kwargs)