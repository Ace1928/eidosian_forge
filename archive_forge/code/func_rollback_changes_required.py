from __future__ import absolute_import, division, print_function
from datetime import datetime
import re
from time import sleep
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.santricity import NetAppESeriesModule
def rollback_changes_required(self):
    """Determine the changes required for snapshot consistency group point-in-time rollback."""
    return self.get_pit_info()