from datetime import datetime, time, timedelta
import pyparsing as pp
import calendar
from_ = CK("from").setParseAction(pp.replaceWith(1))
def remove_temp_keys(t):
    all_keys = list(t.keys())
    for k in all_keys:
        if k not in ('computed_dt', 'original', 'relative_to', 'time_offset', 'calculatedTime'):
            del t[k]