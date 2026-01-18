from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import utils
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.argspec.snmp_server.snmp_server import (
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.rm_templates.snmp_server import (
def get_snmpv3_user_facts(self, snmpv3_user):
    """Parse the snmpv3_user data and return a list of users
        example data-
        User name: TESTU25
        Engine ID: 000000090200000000000A0B
        storage-type: nonvolatile        active access-list: 22
        Authentication Protocol: MD5
        Privacy Protocol: None
        Group-name: TESTG
        :param snmpv3_user: the snmpv3_user data which is a string

        :rtype: list
        :returns: list of users
        """
    user_sets = snmpv3_user.split('User ')
    user_list = []
    re_snmp_auth = re.compile('^Authentication Protocol:\\s*(MD5|SHA)')
    re_snmp_priv = re.compile('^Privacy Protocol:\\s*(3DES|AES|DES)([0-9]*)')
    re_snmp_acl = re.compile('^.*active\\s+(access-list: (\\S+)|)\\s*(IPv6 access-list: (\\S+)|)')
    for user_set in user_sets:
        one_set = {}
        lines = user_set.splitlines()
        for line in lines:
            if line.startswith('name'):
                one_set['username'] = line.split(': ')[1]
                continue
            if line.startswith('Group-name:'):
                one_set['group'] = line.split(': ')[1]
                continue
            re_match = re_snmp_auth.search(line)
            if re_match:
                one_set['authentication'] = {'algorithm': re_match.group(1).lower()}
                continue
            re_match = re_snmp_priv.search(line)
            if re_match:
                one_set['encryption'] = {'priv': re_match.group(1).lower()}
                if re_match.group(2):
                    one_set['encryption']['priv_option'] = re_match.group(2)
                continue
            re_match = re_snmp_acl.search(line)
            if re_match:
                if re_match.group(2):
                    one_set['acl_v4'] = re_match.group(2)
                if re_match.group(4):
                    one_set['acl_v6'] = re_match.group(4)
                continue
            one_set['version'] = 'v3'
        if len(one_set):
            user_list.append(one_set)
    return user_list