import unittest
import macaroonbakery.bakery as bakery
import macaroonbakery.checkers as checkers
from macaroonbakery.tests import common
from pymacaroons.verifier import Verifier
def checker(_, ci):
    caveats = []
    if ci.condition != 'something':
        self.fail('unexpected condition')
    for i in range(0, 2):
        if M.still_required <= 0:
            break
        caveats.append(checkers.Caveat(location=add_bakery(), condition='something'))
        M.still_required -= 1
    return caveats