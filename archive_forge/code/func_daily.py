import datetime
import re
@staticmethod
def daily(t):
    dt = t + datetime.timedelta(days=1)
    return dt.replace(hour=0, minute=0, second=0, microsecond=0)