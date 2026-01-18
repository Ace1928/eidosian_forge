import abc
import os
def put_ops(self, key, time, ops):
    """ Put an ops only if not already there, otherwise it's a no op.
        """
    if self._store.get(key) is None:
        self._store[key] = ops