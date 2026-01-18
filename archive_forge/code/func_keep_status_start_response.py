from webtest import TestApp
from pecan.middleware.recursive import (RecursiveMiddleware,
from pecan.tests import PecanTestCase
def keep_status_start_response(status, headers, exc_info=None):
    return start_response('404 Not Found', headers, exc_info)