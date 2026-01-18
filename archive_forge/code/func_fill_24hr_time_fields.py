from datetime import datetime, time, timedelta
import pyparsing as pp
import calendar
from_ = CK("from").setParseAction(pp.replaceWith(1))
def fill_24hr_time_fields(t):
    t['HH'] = t[0]
    t['MM'] = t[1]
    t['SS'] = 0
    t['ampm'] = ('am', 'pm')[t.HH >= 12]