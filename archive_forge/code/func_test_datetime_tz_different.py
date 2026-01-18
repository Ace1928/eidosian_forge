import datetime
import itertools
from xmlrpc import client as xmlrpclib
import netaddr
from oslotest import base as test_base
from oslo_serialization import msgpackutils
from oslo_utils import uuidutils
def test_datetime_tz_different(self):
    now = datetime.datetime.now()
    if zoneinfo:
        eastern = zoneinfo.ZoneInfo('US/Eastern')
        pacific = zoneinfo.ZoneInfo('US/Pacific')
        now = now.replace(tzinfo=eastern)
        e_dt = now
        now = now.replace(tzinfo=pacific)
        p_dt = now
    else:
        eastern = timezone('US/Eastern')
        pacific = timezone('US/Pacific')
        e_dt = eastern.localize(now)
        p_dt = pacific.localize(now)
    self.assertNotEqual(e_dt, p_dt)
    self.assertNotEqual(e_dt.strftime(_TZ_FMT), p_dt.strftime(_TZ_FMT))
    e_dt2 = _dumps_loads(e_dt)
    p_dt2 = _dumps_loads(p_dt)
    self.assertNotEqual(e_dt2, p_dt2)
    self.assertNotEqual(e_dt2.strftime(_TZ_FMT), p_dt2.strftime(_TZ_FMT))
    self.assertEqual(e_dt, e_dt2)
    self.assertEqual(p_dt, p_dt2)