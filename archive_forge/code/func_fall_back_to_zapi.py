from __future__ import (absolute_import, division, print_function)
from copy import deepcopy
import re
import traceback
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
def fall_back_to_zapi(self, module, msg, parameters):
    if parameters['use_rest'].lower() == 'always':
        module.fail_json(msg='Error: %s' % msg)
    if parameters['use_rest'].lower() == 'auto':
        module.warn('Falling back to ZAPI: %s' % msg)
        return False