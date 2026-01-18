from collections import namedtuple
from datetime import timedelta
from unittest import TestCase
import macaroonbakery.checkers as checkers
import pymacaroons
import pyrfc3339
from pymacaroons import Macaroon
def test_macaroons_expire_time_skips_third_party(self):
    m1 = newMacaroon([checkers.time_before_caveat(t1).condition])
    m2 = newMacaroon()
    m2.add_third_party_caveat('https://example.com', 'a-key', '123')
    t = checkers.macaroons_expiry_time(checkers.Namespace(), [m1, m2])
    self.assertEqual(t1, t)