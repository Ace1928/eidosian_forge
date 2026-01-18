from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_utils import IBMSVCRestApi, svc_argument_spec, get_logger
from ansible.module_utils._text import to_native
def system_update(self, data):
    name_change_required = False
    ntp_change_required = False
    time_change_required = False
    timezone_change_required = False
    tz = (None, None)
    if self.module.check_mode:
        self.changed = True
        return
    if self.systemname and self.systemname != data['name']:
        self.log('Name change detected')
        name_change_required = True
    if self.ntpip and self.ntpip != data['cluster_ntp_IP_address']:
        self.log('NTP change detected')
        ntp_change_required = True
    if self.time and data['cluster_ntp_IP_address'] is not None:
        self.log('TIME change detected, clearing NTP IP')
        ntp_change_required = True
    if self.time:
        self.log('TIME change detected')
        time_change_required = True
    if data['time_zone']:
        tz = data['time_zone'].split(' ', 1)
    if self.timezone and tz[0] != self.timezone:
        timezone_change_required = True
    if name_change_required:
        self.systemname_update()
    if ntp_change_required:
        self.log("updating system properties '%s, %s'", self.systemname, self.ntpip)
        if self.ntpip:
            ip = self.ntpip
        if self.time and ntp_change_required:
            ip = '0.0.0.0'
        self.ntp_update(ip)
    if time_change_required:
        self.systemtime_update()
    if timezone_change_required:
        self.timezone_update()