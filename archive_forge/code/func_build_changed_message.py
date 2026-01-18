from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import vmware_argument_spec, PyVmomi
from ansible.module_utils._text import to_native
def build_changed_message(self, ntp_servers_configured, new_ntp_servers, ntp_servers_to_change, operation):
    """Build changed message"""
    check_mode = 'would be ' if self.module.check_mode else ''
    if operation == 'overwrite':
        add = self.get_not_in_list_one(new_ntp_servers, ntp_servers_configured)
        remove = self.get_not_in_list_one(ntp_servers_configured, new_ntp_servers)
        diff_servers = list(ntp_servers_configured)
        if add and remove:
            for server in add:
                diff_servers.append(server)
            for server in remove:
                diff_servers.remove(server)
            if new_ntp_servers != diff_servers:
                message = 'NTP server %s %sadded and %s %sremoved and the server sequence %schanged as well' % (self.array_to_string(add), check_mode, self.array_to_string(remove), check_mode, check_mode)
            elif new_ntp_servers != ntp_servers_configured:
                message = 'NTP server %s %sreplaced with %s' % (self.array_to_string(remove), check_mode, self.array_to_string(add))
            else:
                message = 'NTP server %s %sremoved and %s %sadded' % (self.array_to_string(remove), check_mode, self.array_to_string(add), check_mode)
        elif add:
            for server in add:
                diff_servers.append(server)
            if new_ntp_servers != diff_servers:
                message = 'NTP server %s %sadded and the server sequence %schanged as well' % (self.array_to_string(add), check_mode, check_mode)
            else:
                message = 'NTP server %s %sadded' % (self.array_to_string(add), check_mode)
        elif remove:
            for server in remove:
                diff_servers.remove(server)
            if new_ntp_servers != diff_servers:
                message = 'NTP server %s %sremoved and the server sequence %schanged as well' % (self.array_to_string(remove), check_mode, check_mode)
            else:
                message = 'NTP server %s %sremoved' % (self.array_to_string(remove), check_mode)
        else:
            message = 'NTP server sequence %schanged' % check_mode
    elif operation == 'add':
        message = 'NTP server %s %sadded' % (self.array_to_string(ntp_servers_to_change), check_mode)
    elif operation == 'delete':
        message = 'NTP server %s %sremoved' % (self.array_to_string(ntp_servers_to_change), check_mode)
    return message