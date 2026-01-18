from __future__ import absolute_import, division, print_function
import functools
from itertools import groupby
from time import sleep
from pprint import pformat
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.netapp import NetAppESeriesModule
from ansible.module_utils._text import to_native
@property
@memoize
def available_drives(self):
    """Determine the list of available drives"""
    return [drive['id'] for drive in self.drives if drive['available'] and drive['status'] == 'optimal']