import sys
import random
import datetime
import simplejson as json
from collections import OrderedDict
def draw_lines(start, total_duration, minute_scale, scale):
    """
    Function to draw the minute line markers and timestamps

    Parameters
    ----------
    start : datetime.datetime obj
        start time for first minute line marker
    total_duration : float
        total duration of the workflow execution (in seconds)
    minute_scale : integer
        the scale, in minutes, at which to plot line markers for the
        gantt chart; for example, minute_scale=10 means there are lines
        drawn at every 10 minute interval from start to finish
    scale : integer
        scale factor in pixel spacing between minute line markers

    Returns
    -------
    result : string
        the html-formatted string for producing the minutes-based
        time line markers
    """
    result = ''
    next_line = 220
    next_time = start
    num_lines = int(total_duration // 60 // minute_scale + 2)
    for line in range(num_lines):
        new_line = "<hr class='line' width='98%%' style='top:%dpx;'>" % next_line
        result += new_line
        time = "<p class='time' style='top:%dpx;'> %02d:%02d </p>" % (next_line - 20, next_time.hour, next_time.minute)
        result += time
        next_line += minute_scale * scale
        next_time += datetime.timedelta(minutes=minute_scale)
    return result