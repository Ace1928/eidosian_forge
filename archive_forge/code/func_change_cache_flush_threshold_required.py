from __future__ import absolute_import, division, print_function
import random
import sys
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.santricity import NetAppESeriesModule, create_multipart_formdata
from ansible.module_utils import six
from ansible.module_utils._text import to_native
def change_cache_flush_threshold_required(self):
    """Determine whether cache flush percentage change is required."""
    if self.cache_flush_threshold is None:
        return False
    current_configuration = self.get_current_configuration()
    if self.cache_flush_threshold <= 0 or self.cache_flush_threshold >= 100:
        self.module.fail_json(msg='Invalid cache flushing threshold, it must be equal to or between 0 and 100. Array [%s]' % self.ssid)
    return self.cache_flush_threshold != current_configuration['cache_settings']['cache_flush_threshold']