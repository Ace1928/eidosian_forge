from dulwich import lru_cache
from dulwich.tests import TestCase
def cleanup_func(key, val):
    cleanup_called.append((key, val))