from __future__ import absolute_import, division, print_function
import functools
from itertools import groupby
from time import sleep
from pprint import pformat
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.netapp import NetAppESeriesModule
from ansible.module_utils._text import to_native
@property
def storage_pool_drives(self):
    """Retrieve list of drives found in storage pool."""
    return [drive for drive in self.drives if drive['currentVolumeGroupRef'] == self.pool_detail['id'] and (not drive['hotSpare'])]