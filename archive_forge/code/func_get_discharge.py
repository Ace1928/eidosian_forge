import os
import unittest
import macaroonbakery.bakery as bakery
import macaroonbakery.checkers as checkers
from macaroonbakery.tests import common
from pymacaroons import MACAROON_V1, Macaroon
def get_discharge(cav, payload):
    return bakery.discharge(common.test_context, cav.caveat_id_bytes, payload, bs.oven.key, common.ThirdPartyStrcmpChecker('true'), bs.oven.locator)