from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from fire import core
from fire import test_components as tc
from fire import testutils
from fire import trace
import mock
import six
@testutils.skipIf(six.PY2, 'lru_cache is Python 3 only.')
def testLruCacheDecoratorBoundArg(self):
    self.assertEqual(core.Fire(tc.py3.LruCacheDecoratedMethod, command=['lru_cache_in_class', 'foo']), 'foo')