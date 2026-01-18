from unittest import mock
from blazarclient import base
from blazarclient import exception
from blazarclient import tests
def test_init_with_url_and_token(self):
    manager = base.BaseClientManager(blazar_url=self.blazar_url, auth_token=self.auth_token, session=None)
    self.assertIsInstance(manager.request_manager, base.RequestManager)