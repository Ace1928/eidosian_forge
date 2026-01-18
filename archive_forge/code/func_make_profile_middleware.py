import sys
import os
import hotshot
import hotshot.stats
import threading
import cgi
import time
from io import StringIO
from paste import response
def make_profile_middleware(app, global_conf, log_filename='profile.log.tmp', limit=40):
    """
    Wrap the application in a component that will profile each
    request.  The profiling data is then appended to the output
    of each page.

    Note that this serializes all requests (i.e., removing
    concurrency).  Therefore never use this in production.
    """
    limit = int(limit)
    return ProfileMiddleware(app, log_filename=log_filename, limit=limit)