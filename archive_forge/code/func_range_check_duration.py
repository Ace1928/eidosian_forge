import calendar
from collections import namedtuple
from aniso8601.exceptions import (
@classmethod
def range_check_duration(cls, PnY=None, PnM=None, PnW=None, PnD=None, TnH=None, TnM=None, TnS=None, rangedict=None):
    if rangedict is None:
        rangedict = cls.DURATION_RANGE_DICT
    if 'PnY' in rangedict:
        PnY = rangedict['PnY'].rangefunc(PnY, rangedict['PnY'])
    if 'PnM' in rangedict:
        PnM = rangedict['PnM'].rangefunc(PnM, rangedict['PnM'])
    if 'PnW' in rangedict:
        PnW = rangedict['PnW'].rangefunc(PnW, rangedict['PnW'])
    if 'PnD' in rangedict:
        PnD = rangedict['PnD'].rangefunc(PnD, rangedict['PnD'])
    if 'TnH' in rangedict:
        TnH = rangedict['TnH'].rangefunc(TnH, rangedict['TnH'])
    if 'TnM' in rangedict:
        TnM = rangedict['TnM'].rangefunc(TnM, rangedict['TnM'])
    if 'TnS' in rangedict:
        TnS = rangedict['TnS'].rangefunc(TnS, rangedict['TnS'])
    return (PnY, PnM, PnW, PnD, TnH, TnM, TnS)