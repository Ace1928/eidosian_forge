import argparse
import os
import sys
import urllib.parse  # noqa: WPS301
from importlib import import_module
from contextlib import suppress
from . import server
from . import wsgi
def parse_wsgi_bind_addr(bind_addr_string):
    """Convert bind address string to bind address parameter."""
    return parse_wsgi_bind_location(bind_addr_string).bind_addr