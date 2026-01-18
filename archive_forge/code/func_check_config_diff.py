from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
def check_config_diff(self):
    """
        Check configuration diff
        Returns: True if there is diff, else False
        """
    iormConfiguration = self.datastore.iormConfiguration
    conf_statsAggregationDisabled = not self.module.params.get('statistic_collection')
    if self.module.params.get('storage_io_control') == 'enable_io_statistics':
        if self.module.params.get('congestion_threshold_manual') is not None:
            conf_congestionThresholdMode = 'manual'
            conf_congestionThreshold = self.module.params.get('congestion_threshold_manual')
            conf_percentOfPeakThroughput = iormConfiguration.percentOfPeakThroughput
        else:
            conf_congestionThresholdMode = 'automatic'
            conf_percentOfPeakThroughput = self.module.params.get('congestion_threshold_percentage')
            conf_congestionThreshold = iormConfiguration.congestionThreshold
        if iormConfiguration.enabled and iormConfiguration.statsCollectionEnabled and (iormConfiguration.statsAggregationDisabled == conf_statsAggregationDisabled) and (iormConfiguration.congestionThresholdMode == conf_congestionThresholdMode) and (iormConfiguration.congestionThreshold == conf_congestionThreshold) and (iormConfiguration.percentOfPeakThroughput == conf_percentOfPeakThroughput):
            return False
        else:
            return True
    elif self.module.params.get('storage_io_control') == 'enable_statistics':
        if not iormConfiguration.enabled and iormConfiguration.statsCollectionEnabled and (iormConfiguration.statsAggregationDisabled == conf_statsAggregationDisabled):
            return False
        else:
            return True
    elif self.module.params.get('storage_io_control') == 'disable':
        if not iormConfiguration.enabled and (not iormConfiguration.statsCollectionEnabled):
            return False
        else:
            return True