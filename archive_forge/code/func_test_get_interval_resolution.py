import unittest
import aniso8601
from aniso8601.builders import (
from aniso8601.exceptions import ISOFormatError
from aniso8601.interval import (
from aniso8601.resolution import IntervalResolution
from aniso8601.tests.compat import mock
def test_get_interval_resolution(self):
    self.assertEqual(_get_interval_resolution(IntervalTuple(start=DateTuple(YYYY='2001', MM='02', DD='03', Www=None, D=None, DDD=None), end=DatetimeTuple(DateTuple(YYYY='2001', MM='02', DD='03', Www=None, D=None, DDD=None), TimeTuple(hh='04', mm='05', ss='06', tz=None)), duration=None)), IntervalResolution.Seconds)
    self.assertEqual(_get_interval_resolution(IntervalTuple(start=DatetimeTuple(DateTuple(YYYY='2001', MM='02', DD='03', Www=None, D=None, DDD=None), TimeTuple(hh='04', mm='05', ss='06', tz=None)), end=DateTuple(YYYY='2001', MM='02', DD='03', Www=None, D=None, DDD=None), duration=None)), IntervalResolution.Seconds)
    self.assertEqual(_get_interval_resolution(IntervalTuple(start=DateTuple(YYYY='2001', MM='02', DD='03', Www=None, D=None, DDD=None), end=None, duration=DurationTuple(PnY='1', PnM='2', PnW=None, PnD='3', TnH='4', TnM='5', TnS='6'))), IntervalResolution.Seconds)
    self.assertEqual(_get_interval_resolution(IntervalTuple(start=DatetimeTuple(DateTuple(YYYY='2001', MM='02', DD='03', Www=None, D=None, DDD=None), TimeTuple(hh='04', mm='05', ss='06', tz=None)), end=None, duration=DurationTuple(PnY='1', PnM='2', PnW=None, PnD='3', TnH=None, TnM=None, TnS=None))), IntervalResolution.Seconds)
    self.assertEqual(_get_interval_resolution(IntervalTuple(start=None, end=DateTuple(YYYY='2001', MM='02', DD='03', Www=None, D=None, DDD=None), duration=DurationTuple(PnY='1', PnM='2', PnW=None, PnD='3', TnH='4', TnM='5', TnS='6'))), IntervalResolution.Seconds)
    self.assertEqual(_get_interval_resolution(IntervalTuple(start=None, end=DatetimeTuple(DateTuple(YYYY='2001', MM='02', DD='03', Www=None, D=None, DDD=None), TimeTuple(hh='04', mm='05', ss='06', tz=None)), duration=DurationTuple(PnY='1', PnM='2', PnW=None, PnD='3', TnH=None, TnM=None, TnS=None))), IntervalResolution.Seconds)