import sys
from pyasn1.type import error
def isSuperTypeOf(self, otherConstraint):
    return otherConstraint is self or not self._values or otherConstraint == self or (self in otherConstraint.getValueMap())