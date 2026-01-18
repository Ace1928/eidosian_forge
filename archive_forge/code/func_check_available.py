import unittest
import cupy.testing._parameterized
def check_available(feature):
    if not is_available():
        raise RuntimeError('cupy.testing: {} is not available.\n\nReason: {}: {}'.format(feature, type(_error).__name__, _error))