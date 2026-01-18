from __future__ import annotations
from dask.rewrite import VAR, RewriteRule, RuleSet, Traverser, args, head
from dask.utils_test import add, inc
def test_RewriteRuleSubs():
    assert rule1.subs({'a': 1}) == (inc, 1)
    assert rule6.subs({'x': [1, 2, 3]}) == [1, 2, 3]