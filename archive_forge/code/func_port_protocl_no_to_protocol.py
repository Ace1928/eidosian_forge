from __future__ import absolute_import, division, print_function
from ansible.module_utils._text import to_text
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.resource_module import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.facts.facts import Facts
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.rm_templates.acls import (
def port_protocl_no_to_protocol(self, num):
    map_protocol = {'179': 'bgp', '19': 'chargen', '514': 'cmd', '13': 'daytime', '9': 'discard', '53': 'domain', '7': 'echo', '512': 'exec', '79': 'finger', '21': 'ftp', '20': 'ftp-data', '70': 'gopher', '101': 'hostname', '113': 'ident', '194': 'irc', '543': 'klogin', '544': 'kshell', '513': 'login', '515': 'lpd', '135': 'msrpc', '119': 'nntp', '5001': 'onep-plain', '5002': 'onep-tls', '496': 'pim-auto-rp', '109': 'pop2', '110': 'pop3', '25': 'smtp', '111': 'sunrpc', '49': 'tacacs', '517': 'talk', '23': 'telnet', '37': 'time', '540': 'uucp', '43': 'whois', '80': 'www'}
    return map_protocol.get(num, num)