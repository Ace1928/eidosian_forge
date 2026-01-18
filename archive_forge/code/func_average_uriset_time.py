import logging
import os
import sys
import threading
import time
import cherrypy
from cherrypy._json import json
def average_uriset_time(s):
    return s['Count'] and s['Sum'] / s['Count'] or 0