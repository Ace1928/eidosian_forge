import logging
import os
import sys
import threading
import time
import cherrypy
from cherrypy._json import json
def get_list_collection(self, v, formatting):
    """Return ([headers], [subrows]) for the given collection."""
    headers = []
    for record in v:
        for k3 in record:
            format = formatting.get(k3, missing)
            if format is None:
                continue
            if k3 not in headers:
                headers.append(k3)
    headers.sort()
    subrows = []
    for record in v:
        subrow = []
        for k3 in headers:
            v3 = record.get(k3, '')
            format = formatting.get(k3, missing)
            if format is None:
                continue
            if hasattr(format, '__call__'):
                v3 = format(v3)
            elif format is not missing:
                v3 = format % v3
            subrow.append(v3)
        subrows.append(subrow)
    return (headers, subrows)