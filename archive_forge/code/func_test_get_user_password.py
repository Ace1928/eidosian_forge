import base64
import sys
import mock
import pytest  # type: ignore
import pyu2f  # type: ignore
from google.auth import exceptions
from google.oauth2 import challenges
def test_get_user_password():
    with mock.patch('getpass.getpass', return_value='foo'):
        assert challenges.get_user_password('') == 'foo'