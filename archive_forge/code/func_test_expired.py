import datetime
import json
import mock
import pytest  # type: ignore
from google.auth import _jwt_async as jwt_async
from google.auth import crypt
from google.auth import exceptions
from tests import test_jwt
def test_expired(self):
    assert not self.credentials.expired
    self.credentials.refresh(None)
    assert not self.credentials.expired
    with mock.patch('google.auth._helpers.utcnow') as now:
        one_day = datetime.timedelta(days=1)
        now.return_value = self.credentials.expiry + one_day
        assert self.credentials.expired