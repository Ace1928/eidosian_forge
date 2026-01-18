from time import localtime
from datetime import date, datetime, time, timedelta
from MySQLdb._mysql import string_literal
def format_TIMEDELTA(v):
    seconds = int(v.seconds) % 60
    minutes = int(v.seconds // 60) % 60
    hours = int(v.seconds // 3600) % 24
    return '%d %d:%d:%d' % (v.days, hours, minutes, seconds)