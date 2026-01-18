from __future__ import absolute_import, division, print_function
import re
from operator import itemgetter
from ansible.module_utils.basic import AnsibleModule
def is_starting_by_ipv6(ip):
    return ipv6_regexp.match(ip) is not None