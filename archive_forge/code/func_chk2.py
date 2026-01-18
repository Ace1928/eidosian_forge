import logging
from io import StringIO
import pytest
from ..batteryrunners import BatteryRunner, Report
def chk2(obj, fix=False):
    rep = Report()
    try:
        ok = obj['testkey'] == 0
    except KeyError:
        rep.problem_level = 20
        rep.problem_msg = 'no "testkey"'
        rep.error = KeyError
        if fix:
            obj['testkey'] = 1
            rep.fix_msg = 'added "testkey"'
        return (obj, rep)
    if ok:
        return (obj, rep)
    rep.problem_level = 10
    rep.problem_msg = '"testkey" != 0'
    rep.error = ValueError
    if fix:
        rep.fix_msg = 'set "testkey" to 0'
        obj['testkey'] = 0
    return (obj, rep)