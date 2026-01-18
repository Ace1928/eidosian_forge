import csv
import datetime
import os
def convert_date(string):
    """Convert a date string in ISO 8601 into a datetime object."""
    if not string:
        date = None
    else:
        parts = [int(x) for x in string.split('-')]
        if len(parts) == 3:
            year, month, day = parts
            date = datetime.date(year, month, day)
        elif len(parts) == 2:
            year, month = parts
            if month == 12:
                date = datetime.date(year, month, 31)
            else:
                date = datetime.date(year, month + 1, 1) - datetime.timedelta(1)
        else:
            raise ValueError('Date not in ISO 8601 format.')
    return date