from __future__ import absolute_import, division, print_function
from datetime import datetime
import re
from time import sleep
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.santricity import NetAppESeriesModule
def create_changes_required(self):
    """Determine the required state changes for creating a new consistency group."""
    changes = {'create_group': {'name': self.group_name, 'alert_threshold_pct': self.alert_threshold_pct, 'maximum_snapshots': self.maximum_snapshots, 'reserve_capacity_full_policy': self.reserve_capacity_full_policy, 'rollback_priority': self.rollback_priority}, 'add_volumes': self.volumes}
    return changes