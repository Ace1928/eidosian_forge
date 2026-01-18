import copy
from datetime import datetime, timedelta
from unittest import TestCase
import macaroonbakery.bakery as bakery
def test_canonical_ops(self):
    canonical_ops_tests = (('empty array', [], []), ('one element', [bakery.Op('a', 'a')], [bakery.Op('a', 'a')]), ('all in order', [bakery.Op('a', 'a'), bakery.Op('a', 'b'), bakery.Op('c', 'c')], [bakery.Op('a', 'a'), bakery.Op('a', 'b'), bakery.Op('c', 'c')]), ('out of order', [bakery.Op('c', 'c'), bakery.Op('a', 'b'), bakery.Op('a', 'a')], [bakery.Op('a', 'a'), bakery.Op('a', 'b'), bakery.Op('c', 'c')]), ('with duplicates', [bakery.Op('c', 'c'), bakery.Op('a', 'b'), bakery.Op('a', 'a'), bakery.Op('c', 'a'), bakery.Op('c', 'b'), bakery.Op('c', 'c'), bakery.Op('a', 'a')], [bakery.Op('a', 'a'), bakery.Op('a', 'b'), bakery.Op('c', 'a'), bakery.Op('c', 'b'), bakery.Op('c', 'c')]), ("make sure we've got the fields right", [bakery.Op(entity='read', action='two'), bakery.Op(entity='read', action='one'), bakery.Op(entity='write', action='one')], [bakery.Op(entity='read', action='one'), bakery.Op(entity='read', action='two'), bakery.Op(entity='write', action='one')]))
    for about, ops, expected in canonical_ops_tests:
        new_ops = copy.copy(ops)
        canonical_ops = bakery.canonical_ops(new_ops)
        self.assertEqual(canonical_ops, expected)
        self.assertEqual(new_ops, ops)