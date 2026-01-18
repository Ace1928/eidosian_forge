from __future__ import absolute_import, division, print_function
import random
import sys
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.santricity import NetAppESeriesModule, create_multipart_formdata
from ansible.module_utils import six
from ansible.module_utils._text import to_native
def change_login_banner_message_required(self):
    """Determine whether storage array name change is required."""
    if self.login_banner_message is None:
        return False
    current_configuration = self.get_current_configuration()
    if self.login_banner_message and sys.getsizeof(self.login_banner_message) > self.MAXIMUM_LOGIN_BANNER_SIZE_BYTES:
        self.module.fail_json(msg='The banner message is too long! It must be %s bytes. Array [%s]' % (self.MAXIMUM_LOGIN_BANNER_SIZE_BYTES, self.ssid))
    return self.login_banner_message != current_configuration['login_banner_message']