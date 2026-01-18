from collections import namedtuple
from datetime import timedelta
from unittest import TestCase
import macaroonbakery.checkers as checkers
import pymacaroons
import pyrfc3339
from pymacaroons import Macaroon
def newMacaroon(conds=[]):
    m = Macaroon(key='key', version=2)
    for cond in conds:
        m.add_first_party_caveat(cond)
    return m