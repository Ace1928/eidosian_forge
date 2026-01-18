import base64
import os
import stat
from cryptography import fernet
from oslo_log import log
from keystone.common import utils
import keystone.conf
def create_key_directory(self, keystone_user_id=None, keystone_group_id=None):
    """Attempt to create the key directory if it doesn't exist."""
    utils.create_directory(self.key_repository, keystone_user_id=keystone_user_id, keystone_group_id=keystone_group_id)