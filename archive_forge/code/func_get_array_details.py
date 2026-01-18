from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell \
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell.libraries.configuration \
import re
def get_array_details(self):
    """ Get system details of a powerflex array """
    try:
        LOG.info('Getting array details ')
        entity_list = ['addressSpaceUsage', 'authenticationMethod', 'capacityAlertCriticalThresholdPercent', 'capacityAlertHighThresholdPercent', 'capacityTimeLeftInDays', 'cliPasswordAllowed', 'daysInstalled', 'defragmentationEnabled', 'enterpriseFeaturesEnabled', 'id', 'installId', 'isInitialLicense', 'lastUpgradeTime', 'managementClientSecureCommunicationEnabled', 'maxCapacityInGb', 'mdmCluster', 'mdmExternalPort', 'mdmManagementPort', 'mdmSecurityPolicy', 'showGuid', 'swid', 'systemVersionName', 'tlsVersion', 'upgradeState']
        sys_list = self.powerflex_conn.system.get()
        sys_details_list = []
        for sys in sys_list:
            sys_details = {}
            for entity in entity_list:
                if entity in sys.keys():
                    sys_details.update({entity: sys[entity]})
            if sys_details:
                sys_details_list.append(sys_details)
        return sys_details_list
    except Exception as e:
        msg = 'Get array details from Powerflex array failed with error %s' % str(e)
        LOG.error(msg)
        self.module.fail_json(msg=msg)