import datetime
import mock
import pytest  # type: ignore
from google.auth import app_engine
@mock.patch.object(app_engine, 'app_identity', new=None)
def test_missing_apis(self):
    with pytest.raises(EnvironmentError) as excinfo:
        app_engine.Credentials()
    assert excinfo.match('App Engine APIs are not available')