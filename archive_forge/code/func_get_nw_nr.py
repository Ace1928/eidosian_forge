from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
import os
import re
def get_nw_nr(sids, module):
    nw_list = list()
    type = ''
    for sid in sids:
        for instance in os.listdir('/usr/sap/' + sid):
            instance_nr = instance[-2:]
            command = [module.get_bin_path('/usr/sap/hostctrl/exe/sapcontrol', required=True)]
            if instance_nr.isdigit():
                command.extend(['-nr', instance_nr, '-function', 'GetInstanceProperties'])
                check_instance = module.run_command(command, check_rc=False)
                if check_instance[0] != 1:
                    for line in check_instance[1].splitlines():
                        if re.search('INSTANCE_NAME', line):
                            type_raw = line.strip('][').split(', ')[-1]
                            type = type_raw[:-2]
                            nw_list.append({'NR': instance_nr, 'SID': sid, 'TYPE': get_instance_type(type), 'InstanceType': 'NW'})
    return nw_list