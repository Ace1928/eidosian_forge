from __future__ import absolute_import, division, print_function
import re
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
def convert_host_str_to_list(self, host_str):
    """ Convert host_str which have comma separated hosts to host_list with
            ip4/ip6 host obj if IP4/IP6 like string found

        :param host_str: hosts str separated by comma
        :return: hosts list, which may contains IP4/IP6 object if given in
                host_str
        :rytpe: list
        """
    if not host_str:
        LOG.debug('Empty host_str given')
        return []
    host_list = []
    try:
        for h in host_str.split(','):
            version = get_ip_version(h)
            if version == 4:
                h = u'{0}'.format(h)
                h = IPv4Network(h, strict=False)
            elif version == 6:
                h = u'{0}'.format(h)
                h = IPv6Network(h, strict=False)
            host_list.append(h)
    except Exception as e:
        msg = 'Error while converting host_str: %s to list error: %s' % (host_str, str(e))
        LOG.error(msg)
        self.module.fail_json(msg=msg)
    return host_list