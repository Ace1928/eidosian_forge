from __future__ import absolute_import, division, print_function
import re
from copy import deepcopy
from re import M, findall, search
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import utils
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.argspec.firewall_rules.firewall_rules import (
def parse_tcp(self, conf, attrib=None):
    """
        This function triggers the parsing of 'tcp' attributes.
        :param conf: configuration.
        :param attrib: 'tcp'.
        :return: generated config dictionary.
        """
    cfg_dict = self.parse_attr(conf, ['flags'], match=attrib)
    return cfg_dict