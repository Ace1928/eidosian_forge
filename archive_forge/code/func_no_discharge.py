import unittest
import macaroonbakery.bakery as bakery
import macaroonbakery.checkers as checkers
from macaroonbakery.tests import common
from pymacaroons.verifier import Verifier
def no_discharge(test):

    def get_discharge(cav, payload):
        test.fail('get_discharge called unexpectedly')
    return get_discharge