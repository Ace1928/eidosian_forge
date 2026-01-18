from __future__ import (absolute_import, division, print_function)
import json
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible.module_utils.urls import ConnectionError
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
def baseline_based_update(rest_obj, module, baseline, dev_comp_map):
    compliance_uri = COMPLIANCE_URI.format(baseline['baseline_id'])
    resp = rest_obj.get_all_report_details(compliance_uri)
    compliance_report_list = []
    update_actions = ['UPGRADE', 'DOWNGRADE']
    if resp['report_list']:
        comps = []
        if not dev_comp_map:
            comps = module.params.get('components')
            dev_comp_map = dict([(str(dev['DeviceId']), comps) for dev in resp['report_list']])
        for dvc in resp['report_list']:
            dev_id = dvc['DeviceId']
            if str(dev_id) in dev_comp_map:
                comps = dev_comp_map.get(str(dev_id), [])
                compliance_report = dvc.get('ComponentComplianceReports')
                if compliance_report is not None:
                    data_dict = {}
                    comp_list = []
                    if not comps:
                        comp_list = list((icomp['SourceName'] for icomp in compliance_report if icomp['UpdateAction'] in update_actions))
                    else:
                        comp_list = list((icomp['SourceName'] for icomp in compliance_report if icomp['UpdateAction'] in update_actions and icomp.get('Name') in comps))
                    if comp_list:
                        data_dict['Id'] = dev_id
                        data_dict['Data'] = str(';').join(comp_list)
                        data_dict['TargetType'] = {'Id': dvc['DeviceTypeId'], 'Name': dvc['DeviceTypeName']}
                        compliance_report_list.append(data_dict)
    else:
        module.fail_json(msg=COMPLIANCE_READ_FAIL)
    if not compliance_report_list:
        module.exit_json(msg=NO_CHANGES_MSG)
    if module.check_mode:
        module.exit_json(msg=CHANGES_FOUND, changed=True)
    return compliance_report_list