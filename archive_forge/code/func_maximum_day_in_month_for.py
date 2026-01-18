import calendar
from datetime import datetime
from datetime import timedelta
import re
import sys
import time
def maximum_day_in_month_for(year, month):
    return calendar.monthrange(year, month)[1]