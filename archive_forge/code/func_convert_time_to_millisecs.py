from __future__ import absolute_import, division, print_function
def convert_time_to_millisecs(time):
    """Convert a time period in milliseconds"""
    if time[-1:].lower() not in ['w', 'd', 'h', 'm', 's']:
        return 0
    try:
        if time[-1:].lower() == 'w':
            return int(time[:-1]) * 7 * 86400000
        elif time[-1:].lower() == 'd':
            return int(time[:-1]) * 86400000
        elif time[-1:].lower() == 'h':
            return int(time[:-1]) * 3600000
        elif time[-1:].lower() == 'm':
            return int(time[:-1]) * 60000
    except Exception:
        return 0