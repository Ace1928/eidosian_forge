from __future__ import print_function
import six
from patsy import PatsyError
from patsy.parse_formula import ParseNode, Token, parse_formula
from patsy.eval import EvalEnvironment, EvalFactor
from patsy.util import uniqueify_list
from patsy.util import repr_pretty_delegate, repr_pretty_impl
from patsy.util import no_pickling, assert_no_pickling
def test_ModelDesc():
    f1 = _MockFactor('a')
    f2 = _MockFactor('b')
    m = ModelDesc([INTERCEPT, Term([f1])], [Term([f1]), Term([f1, f2])])
    assert m.lhs_termlist == [INTERCEPT, Term([f1])]
    assert m.rhs_termlist == [Term([f1]), Term([f1, f2])]
    print(m.describe())
    assert m.describe() == '1 + a ~ 0 + a + a:b'
    assert_no_pickling(m)
    assert ModelDesc([], []).describe() == '~ 0'
    assert ModelDesc([INTERCEPT], []).describe() == '1 ~ 0'
    assert ModelDesc([INTERCEPT], [INTERCEPT]).describe() == '1 ~ 1'
    assert ModelDesc([INTERCEPT], [INTERCEPT, Term([f2])]).describe() == '1 ~ b'