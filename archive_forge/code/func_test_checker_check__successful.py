from unittest import mock
import requests
from update_checker import UpdateChecker, update_check
@mock.patch('requests.get')
def test_checker_check__successful(mock_get):
    mock_response(mock_get.return_value)
    checker = UpdateChecker(bypass_cache=True)
    result = checker.check(PACKAGE, '1.0.0')
    assert result.available_version == '5.0.0'