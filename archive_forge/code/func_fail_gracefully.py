import functools
import sys
import flask
import oslo_i18n
from oslo_log import log
from oslo_middleware import healthcheck
import keystone.api
from keystone import exception
from keystone.oauth2 import handlers as oauth2_handlers
from keystone.receipt import handlers as receipt_handlers
from keystone.server.flask import common as ks_flask
from keystone.server.flask.request_processing import json_body
from keystone.server.flask.request_processing import req_logging
def fail_gracefully(f):
    """Log exceptions and aborts."""

    @functools.wraps(f)
    def wrapper(*args, **kw):
        try:
            return f(*args, **kw)
        except Exception as e:
            LOG.debug(e, exc_info=True)
            LOG.critical(e)
            sys.exit(1)
    return wrapper