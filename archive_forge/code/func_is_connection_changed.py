from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_text
import re
def is_connection_changed(self):
    options = {'connection.interface-name': self.ifname}
    if self.type == 'vpn' and self.ifname is None:
        del options['connection.interface-name']
    if not self.type:
        current_con_type = self.show_connection().get('connection.type')
        if current_con_type:
            if current_con_type == '802-11-wireless':
                current_con_type = 'wifi'
            self.type = current_con_type
    options.update(self.connection_options(detect_change=True))
    return self._compare_conn_params(self.show_connection(), options)