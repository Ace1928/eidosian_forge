import logging
from openstack.config import exceptions as sdk_exceptions
from openstack.config import loader as config
from oslo_utils import strutils
def load_auth_plugin(self, config):
    """Get auth plugin and validate args"""
    loader = self._get_auth_loader(config)
    config = self._validate_auth(config, loader)
    auth_plugin = loader.load_from_options(**config['auth'])
    return auth_plugin