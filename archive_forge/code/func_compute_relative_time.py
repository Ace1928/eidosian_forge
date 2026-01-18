from datetime import datetime, time, timedelta
import pyparsing as pp
import calendar
from_ = CK("from").setParseAction(pp.replaceWith(1))
def compute_relative_time(t):
    if 'ref_time' not in t:
        t['ref_time'] = datetime.now().time().replace(microsecond=0)
    else:
        t['ref_time'] = t.ref_time.computed_time
    delta_seconds = {'hour': 3600, 'minute': 60, 'second': 1}[t.units] * t.qty
    t['time_delta'] = timedelta(seconds=t.dir * delta_seconds)