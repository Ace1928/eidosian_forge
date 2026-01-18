from datetime import datetime, time, timedelta
import pyparsing as pp
import calendar
from_ = CK("from").setParseAction(pp.replaceWith(1))
def fill_default_time_fields(t):
    for fld in 'HH MM SS'.split():
        if fld not in t:
            t[fld] = 0