import argparse
import json
import logging
import socket
import threading
from http.server import HTTPServer, SimpleHTTPRequestHandler
from ray.autoscaler._private.local.node_provider import LocalNodeProvider
def do_HEAD(self):
    """HTTP HEAD handler method."""
    self._do_header()