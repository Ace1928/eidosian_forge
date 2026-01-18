from collections import namedtuple
from datetime import timedelta
from unittest import TestCase
import macaroonbakery.checkers as checkers
import pymacaroons
import pyrfc3339
from pymacaroons import Macaroon
def test_macaroons_expire_time(self):
    ExpireTest = namedtuple('ExpireTest', 'about macaroons expectTime')
    tests = [ExpireTest(about='no macaroons', macaroons=[newMacaroon()], expectTime=None), ExpireTest(about='single macaroon without caveats', macaroons=[newMacaroon()], expectTime=None), ExpireTest(about='multiple macaroon without caveats', macaroons=[newMacaroon()], expectTime=None), ExpireTest(about='single macaroon with time-before caveat', macaroons=[newMacaroon([checkers.time_before_caveat(t1).condition])], expectTime=t1), ExpireTest(about='single macaroon with multiple time-before caveats', macaroons=[newMacaroon([checkers.time_before_caveat(t2).condition, checkers.time_before_caveat(t1).condition])], expectTime=t1), ExpireTest(about='multiple macaroons with multiple time-before caveats', macaroons=[newMacaroon([checkers.time_before_caveat(t3).condition, checkers.time_before_caveat(t1).condition]), newMacaroon([checkers.time_before_caveat(t3).condition, checkers.time_before_caveat(t1).condition])], expectTime=t1)]
    for test in tests:
        print('test ', test.about)
        t = checkers.macaroons_expiry_time(checkers.Namespace(), test.macaroons)
        self.assertEqual(t, test.expectTime)