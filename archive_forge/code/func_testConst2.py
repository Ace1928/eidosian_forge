import sys
from tests.base import BaseTestCase
from pyasn1.type import constraint
from pyasn1.type import error
def testConst2(self):
    c = constraint.InnerTypeConstraint((0, constraint.SingleValueConstraint(4), 'PRESENT'), (1, constraint.SingleValueConstraint(4), 'ABSENT'))
    try:
        c(4, 0)
    except error.ValueConstraintError:
        raise
        assert 0, 'constraint check fails'
    try:
        c(4, 1)
    except error.ValueConstraintError:
        pass
    else:
        assert 0, 'constraint check fails'
    try:
        c(3, 0)
    except error.ValueConstraintError:
        pass
    else:
        assert 0, 'constraint check fails'