from unittest import mock
import requests
from update_checker import UpdateChecker, update_check
@mock.patch('requests.get')
def test_checker_check__no_update_to_beta_version(mock_get):
    mock_response(mock_get.return_value, '3.7.0b1')
    checker = UpdateChecker(bypass_cache=True)
    assert checker.check(PACKAGE, '3.6') is None