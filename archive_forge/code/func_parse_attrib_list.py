from __future__ import absolute_import, division, print_function
from copy import deepcopy
from re import M, findall, search
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import utils
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.argspec.ospfv2.ospfv2 import (
def parse_attrib_list(self, conf, attrib, param):
    """
        This function forms the regex to fetch the listed attributes
        from config
        :param conf: configuration data
        :param attrib: attribute name
        :param param: parameter data
        :return: generated rule list configuration
        """
    r_lst = []
    if attrib == 'area':
        items = findall('^' + attrib.replace('_', '-') + " (?:'*)(\\S+)(?:'*)", conf, M)
    elif attrib == 'key-id':
        items = findall('^.*' + attrib.replace('_', '-') + " (?:'*)(\\S+)(?:'*)", conf, M)
    else:
        items = findall('' + attrib + " (?:'*)(\\S+)(?:'*)", conf, M)
    if items:
        a_lst = []
        for item in set(items):
            i_regex = ' %s .+$' % item
            cfg = '\n'.join(findall(i_regex, conf, M))
            if attrib == 'area':
                obj = self.parse_area(cfg, item)
            elif attrib == 'virtual-link':
                obj = self.parse_vlink(cfg)
            elif attrib == 'key-id':
                obj = self.parse_key(cfg, item)
            else:
                obj = self.parse_attrib(cfg, attrib)
            obj[param] = item.strip("'")
            if obj:
                a_lst.append(obj)
        r_lst = sorted(a_lst, key=lambda i: i[param])
    return r_lst