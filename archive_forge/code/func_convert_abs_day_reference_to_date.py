from datetime import datetime, time, timedelta
import pyparsing as pp
import calendar
from_ = CK("from").setParseAction(pp.replaceWith(1))
def convert_abs_day_reference_to_date(t):
    now = datetime.now().replace(microsecond=0)
    if 'day_name' in t:
        todaynum = now.weekday()
        daynames = [n.lower() for n in weekday_name_list]
        nameddaynum = daynames.index(t.day_name.lower())
        if t.dir > 0:
            daydiff = (nameddaynum + 7 - todaynum) % 7 or 7
        else:
            daydiff = -((todaynum + 7 - nameddaynum) % 7 or 7)
        t['abs_date'] = datetime(now.year, now.month, now.day) + timedelta(daydiff)
    else:
        name = t[0]
        t['abs_date'] = {'now': now, 'today': datetime(now.year, now.month, now.day), 'yesterday': datetime(now.year, now.month, now.day) + timedelta(days=-1), 'tomorrow': datetime(now.year, now.month, now.day) + timedelta(days=+1)}[name]