import sys
import os
import hotshot
import hotshot.stats
import threading
import cgi
import time
from io import StringIO
from paste import response
def replace_start_response(status, headers, exc_info=None):
    catch_response.extend([status, headers])
    start_response(status, headers, exc_info)
    return body.append