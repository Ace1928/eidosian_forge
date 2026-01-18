from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
import os
import re
def get_hana_nr(sids, module):
    hana_list = list()
    for sid in sids:
        for instance in os.listdir('/usr/sap/' + sid):
            if 'HDB' in instance:
                instance_nr = instance[-2:]
                command = [module.get_bin_path('/usr/sap/hostctrl/exe/sapcontrol', required=True)]
                command.extend(['-nr', instance_nr, '-function', 'GetProcessList'])
                check_instance = module.run_command(command, check_rc=False)
                if check_instance[0] != 1:
                    hana_list.append({'NR': instance_nr, 'SID': sid, 'TYPE': 'HDB', 'InstanceType': 'HANA'})
    return hana_list