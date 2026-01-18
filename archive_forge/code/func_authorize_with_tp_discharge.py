import base64
import json
from collections import namedtuple
from datetime import timedelta
from unittest import TestCase
import macaroonbakery.bakery as bakery
import macaroonbakery.checkers as checkers
import pymacaroons
from macaroonbakery.tests.common import epoch, test_checker, test_context
from pymacaroons.verifier import FirstPartyCaveatVerifierDelegate, Verifier
def authorize_with_tp_discharge(ctx, id, op):
    if id is not None and id.id() == 'bob' and (op == bakery.Op(entity='something', action='read')):
        return (True, [checkers.Caveat(condition='question', location='other third party')])
    return (False, None)