import testtools
from unittest import mock
from troveclient.v1 import limits
def test_list_errors(self):
    status_list = [400, 401, 403, 404, 408, 409, 413, 500, 501]
    for status_code in status_list:
        self._check_error_response(status_code)