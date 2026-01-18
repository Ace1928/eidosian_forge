import datetime
import io
import logging
import os
import re
import subprocess
import sys
import time
import unittest
import warnings
import contextlib
import portend
import pytest
from cheroot.test import webtest
import cherrypy
from cherrypy._cpcompat import text_or_bytes, HTTPSConnection, ntob
from cherrypy.lib import httputil
from cherrypy.lib import gctools
def get_modfastcgi_supervisor(**options):
    from cherrypy.test import modfastcgi
    return modfastcgi.ModFCGISupervisor(**options)