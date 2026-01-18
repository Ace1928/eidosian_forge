from tornado import web
from jupyter_server.auth.decorator import authorized
from ...base.handlers import APIHandler
from . import csp_report_uri
def skip_check_origin(self):
    """Don't check origin when reporting origin-check violations!"""
    return True