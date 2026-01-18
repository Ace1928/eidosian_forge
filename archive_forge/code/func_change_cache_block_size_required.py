from __future__ import absolute_import, division, print_function
import random
import sys
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.santricity import NetAppESeriesModule, create_multipart_formdata
from ansible.module_utils import six
from ansible.module_utils._text import to_native
def change_cache_block_size_required(self):
    """Determine whether cache block size change is required."""
    if self.cache_block_size is None:
        return False
    current_configuration = self.get_current_configuration()
    current_available_block_sizes = current_configuration['cache_block_size_options']
    if self.cache_block_size not in current_available_block_sizes:
        self.module.fail_json(msg='Invalid cache block size. Array [%s]. Available cache block sizes [%s].' % (self.ssid, current_available_block_sizes))
    return self.cache_block_size != current_configuration['cache_settings']['cache_block_size']