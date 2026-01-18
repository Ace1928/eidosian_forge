import os, sys, time, re, calendar
import py
import subprocess
from py._path import common
def parse_apr_time(timestr):
    i = timestr.rfind('.')
    if i == -1:
        raise ValueError('could not parse %s' % timestr)
    timestr = timestr[:i]
    parsedtime = time.strptime(timestr, '%Y-%m-%dT%H:%M:%S')
    return time.mktime(parsedtime)