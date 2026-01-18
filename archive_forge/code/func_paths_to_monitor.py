from __future__ import print_function
import logging
import os
import sys
import threading
import time
import subprocess
from wsgiref.simple_server import WSGIRequestHandler
from pecan.commands import BaseCommand
from pecan import util
def paths_to_monitor(self, conf):
    paths = []
    for package_name in getattr(conf.app, 'modules', []):
        module = __import__(package_name, fromlist=['app'])
        if hasattr(module, 'app') and hasattr(module.app, 'setup_app'):
            paths.append((os.path.dirname(module.__file__), True))
            break
    paths.append((os.path.dirname(conf.__file__), False))
    return paths