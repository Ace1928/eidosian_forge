import os
import pickle
import re
import requests
import sys
import time
from datetime import datetime
from functools import wraps
from tempfile import gettempdir
def pretty_date(the_datetime):
    """Attempt to return a human-readable time delta string."""
    diff = datetime.utcnow() - the_datetime
    if diff.days > 7 or diff.days < 0:
        return the_datetime.strftime('%A %B %d, %Y')
    elif diff.days == 1:
        return '1 day ago'
    elif diff.days > 1:
        return f'{diff.days} days ago'
    elif diff.seconds <= 1:
        return 'just now'
    elif diff.seconds < 60:
        return f'{diff.seconds} seconds ago'
    elif diff.seconds < 120:
        return '1 minute ago'
    elif diff.seconds < 3600:
        return f'{int(round(diff.seconds / 60))} minutes ago'
    elif diff.seconds < 7200:
        return '1 hour ago'
    else:
        return f'{int(round(diff.seconds / 3600))} hours ago'