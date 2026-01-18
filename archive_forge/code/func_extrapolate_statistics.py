import logging
import os
import sys
import threading
import time
import cherrypy
from cherrypy._json import json
def extrapolate_statistics(scope):
    """Return an extrapolated copy of the given scope."""
    c = {}
    for k, v in scope.copy().items():
        if isinstance(v, dict):
            v = extrapolate_statistics(v)
        elif isinstance(v, (list, tuple)):
            v = [extrapolate_statistics(record) for record in v]
        elif hasattr(v, '__call__'):
            v = v(scope)
        c[k] = v
    return c