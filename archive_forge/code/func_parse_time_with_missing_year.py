import os, sys, time, re
import py
from py import path, process
from py._path import common
from py._path import svnwc as svncommon
from py._path.cacheutil import BuildcostAccessCache, AgingCache
def parse_time_with_missing_year(timestr):
    """ analyze the time part from a single line of "svn ls -v"
    the svn output doesn't show the year makes the 'timestr'
    ambigous.
    """
    import calendar
    t_now = time.gmtime()
    tparts = timestr.split()
    month = time.strptime(tparts.pop(0), '%b')[1]
    day = time.strptime(tparts.pop(0), '%d')[2]
    last = tparts.pop(0)
    try:
        if ':' in last:
            raise ValueError()
        year = time.strptime(last, '%Y')[0]
        hour = minute = 0
    except ValueError:
        hour, minute = time.strptime(last, '%H:%M')[3:5]
        year = t_now[0]
        t_result = (year, month, day, hour, minute, 0, 0, 0, 0)
        if t_result > t_now:
            year -= 1
    t_result = (year, month, day, hour, minute, 0, 0, 0, 0)
    return calendar.timegm(t_result)