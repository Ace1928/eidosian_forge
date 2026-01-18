import flask
from oslo_log import log
def log_request_info():
    LOG.debug('REQUEST_METHOD: `%s`', flask.request.method)
    LOG.debug('SCRIPT_NAME: `%s`', flask.request.script_root)
    LOG.debug('PATH_INFO: `%s`', flask.request.path)