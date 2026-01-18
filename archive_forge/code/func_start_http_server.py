import errno
import functools
import http.client
import http.server
import io
import os
import shlex
import shutil
import signal
import socket
import subprocess
import threading
import time
from unittest import mock
from alembic import command as alembic_command
import fixtures
from oslo_concurrency import lockutils
from oslo_config import cfg
from oslo_config import fixture as cfg_fixture
from oslo_log.fixture import logging_error as log_fixture
from oslo_log import log
from oslo_utils import timeutils
from oslo_utils import units
import testtools
import webob
from glance.api.v2 import cached_images
from glance.common import config
from glance.common import exception
from glance.common import property_utils
from glance.common import utils
from glance.common import wsgi
from glance import context
from glance.db.sqlalchemy import alembic_migrations
from glance.db.sqlalchemy import api as db_api
from glance.tests.unit import fixtures as glance_fixtures
def start_http_server(image_id, image_data):

    def _get_http_handler_class(fixture):

        class StaticHTTPRequestHandler(http.server.BaseHTTPRequestHandler):

            def do_GET(self):
                self.send_response(http.client.OK)
                self.send_header('Content-Length', str(len(fixture)))
                self.end_headers()
                self.wfile.write(fixture.encode('latin-1'))
                return

            def do_HEAD(self):
                if 'non_existing_image_path' in self.path:
                    self.send_response(http.client.NOT_FOUND)
                else:
                    self.send_response(http.client.OK)
                self.send_header('Content-Length', str(len(fixture)))
                self.end_headers()
                return

            def log_message(self, *args, **kwargs):
                return
        return StaticHTTPRequestHandler
    server_address = ('127.0.0.1', 0)
    handler_class = _get_http_handler_class(image_data)
    httpd = http.server.HTTPServer(server_address, handler_class)
    port = httpd.socket.getsockname()[1]
    thread = threading.Thread(target=httpd.serve_forever)
    thread.daemon = True
    thread.start()
    return (thread, httpd, port)