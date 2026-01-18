import base64
import os
import stat
from cryptography import fernet
from oslo_log import log
from keystone.common import utils
import keystone.conf
def initialize_key_repository(self, keystone_user_id=None, keystone_group_id=None):
    """Create a key repository and bootstrap it with a key.

        :param keystone_user_id: User ID of the Keystone user.
        :param keystone_group_id: Group ID of the Keystone user.

        """
    if os.access(os.path.join(self.key_repository, '0'), os.F_OK):
        LOG.info('Key repository is already initialized; aborting.')
        return
    self._create_new_key(keystone_user_id, keystone_group_id)
    self.rotate_keys(keystone_user_id, keystone_group_id)