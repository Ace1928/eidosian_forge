import logging
from io import StringIO
import pytest
from ..batteryrunners import BatteryRunner, Report
def chk_error(obj, fix=False):
    rep = Report(KeyError)
    if not 'thirdkey' in obj:
        rep.problem_level = 40
        rep.problem_msg = 'no "thirdkey"'
        if fix:
            obj['anotherkey'] = 'a string'
            rep.fix_msg = 'added "anotherkey"'
    return (obj, rep)