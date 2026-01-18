from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.nxos import (
def get_igmp_snooping(module):
    command = 'show ip igmp snooping'
    existing = {}
    try:
        body = execute_show_command(command, module, output='json')[0]
    except IndexError:
        body = []
    if body:
        snooping = str(body.get('enabled')).lower()
        if 'none' in snooping:
            snooping = str(body.get('GlobalSnoopEnabled')).lower()
        if snooping == 'true' or snooping == 'enabled' or snooping == 'yes':
            existing['snooping'] = True
        else:
            existing['snooping'] = False
        report_supp = str(body.get('grepsup')).lower()
        if 'none' in report_supp:
            report_supp = str(body.get('GlobalReportSupression')).lower()
        if report_supp == 'true' or report_supp == 'enabled':
            existing['report_supp'] = True
        else:
            existing['report_supp'] = False
        link_local_grp_supp = str(body.get('glinklocalgrpsup')).lower()
        if 'none' in link_local_grp_supp:
            link_local_grp_supp = str(body.get('GlobalLinkLocalGroupSupression')).lower()
        if link_local_grp_supp == 'true' or link_local_grp_supp == 'enabled':
            existing['link_local_grp_supp'] = True
        else:
            existing['link_local_grp_supp'] = False
        v3_report_supp = str(body.get('gv3repsup')).lower()
        if 'none' in v3_report_supp:
            v3_report_supp = str(body.get('GlobalV3ReportSupression')).lower()
        if v3_report_supp == 'true' or v3_report_supp == 'enabled':
            existing['v3_report_supp'] = True
        else:
            existing['v3_report_supp'] = False
    command = 'show ip igmp snooping'
    body = execute_show_command(command, module)[0]
    if body:
        existing['group_timeout'] = get_group_timeout(body)
    return existing