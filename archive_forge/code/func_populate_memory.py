from __future__ import absolute_import, division, print_function
import platform
import re
from ansible.module_utils.six import iteritems
from ansible_collections.arista.eos.plugins.module_utils.network.eos.eos import (
def populate_memory(self):
    values = self.responses[1]
    return dict(memfree_mb=int(values['memFree']) / 1024, memtotal_mb=int(values['memTotal']) / 1024)