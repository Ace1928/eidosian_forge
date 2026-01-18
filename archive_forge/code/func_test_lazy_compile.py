import pickle
import re
from typing import List, Tuple
from .. import lazy_regex, tests
def test_lazy_compile(self):
    """Make sure that LazyRegex objects compile at the right time"""
    actions = []
    InstrumentedLazyRegex.use_actions(actions)
    pattern = InstrumentedLazyRegex(args=('foo',), kwargs={})
    actions.append(('created regex', 'foo'))
    pattern.match('foo')
    pattern.match('foo')
    self.assertEqual([('created regex', 'foo'), ('__getattr__', 'match'), ('_real_re_compile', ('foo',), {})], actions)