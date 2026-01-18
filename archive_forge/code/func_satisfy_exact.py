import binascii
from pymacaroons.binders import HashSignaturesBinder
from pymacaroons.exceptions import MacaroonInvalidSignatureException
from pymacaroons.caveat_delegates import (
from pymacaroons.utils import (
def satisfy_exact(self, predicate):
    if predicate is None:
        raise TypeError('Predicate cannot be none.')
    self.predicates.append(convert_to_string(predicate))