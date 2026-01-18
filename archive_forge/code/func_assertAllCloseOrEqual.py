import collections
import numpy as np
import tensorflow.compat.v2 as tf
def assertAllCloseOrEqual(self, a, b, msg=None):
    """Asserts that elements are close (if numeric) or equal (if string)."""
    if a is None or b is None:
        self.assertAllEqual(a, b, msg=msg)
    elif isinstance(a, (list, tuple)):
        self.assertEqual(len(a), len(b))
        for a_value, b_value in zip(a, b):
            self.assertAllCloseOrEqual(a_value, b_value, msg=msg)
    elif isinstance(a, collections.abc.Mapping):
        self.assertEqual(len(a), len(b))
        for key, a_value in a.items():
            b_value = b[key]
            error_message = f'{msg} ({key})' if msg else None
            self.assertAllCloseOrEqual(a_value, b_value, error_message)
    elif isinstance(a, float) or (hasattr(a, 'dtype') and np.issubdtype(a.dtype, np.number)):
        self.assertAllClose(a, b, msg=msg)
    else:
        self.assertAllEqual(a, b, msg=msg)