from __future__ import absolute_import, division, print_function
import re
import time
import traceback
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.network import NetworkModule, NetworkError
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.config import NetworkConfig, dumps
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.parsing import Command
from ansible_collections.community.network.plugins.module_utils.network.ordnance.ordnance import get_config
from ansible.module_utils.six import iteritems
from ansible.module_utils._text import to_native
def load_banners(module, banners):
    delimiter = module.params['multiline_delimiter']
    for key, value in iteritems(banners):
        key += ' %s' % delimiter
        for cmd in ['config terminal', key, value, delimiter, 'end']:
            cmd += '\r'
            module.connection.shell.shell.sendall(cmd)
        time.sleep(1)
        module.connection.shell.receive()