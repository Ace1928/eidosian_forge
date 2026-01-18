import logging
import os
import sys
import threading
import time
import cherrypy
from cherrypy._json import json
def locale_date(v):
    return time.strftime('%c', time.gmtime(v))