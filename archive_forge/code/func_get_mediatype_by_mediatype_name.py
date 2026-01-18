from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.zabbix.plugins.module_utils.base import ZabbixBase
from ansible.module_utils.compat.version import LooseVersion
import ansible_collections.community.zabbix.plugins.module_utils.helpers as zabbix_utils
def get_mediatype_by_mediatype_name(self, mediatype_name):
    """Get mediatype by mediatype name

        Args:
            mediatype_name: mediatype name

        Returns:
            mediatype matching mediatype name

        """
    filter = {'name': [mediatype_name]}
    try:
        if str(mediatype_name).lower() == 'all':
            return '0'
        mediatype_list = self._zapi.mediatype.get({'output': 'extend', 'filter': filter})
        if len(mediatype_list) < 1:
            self._module.fail_json(msg='Media type not found: %s' % mediatype_name)
        else:
            return mediatype_list[0]['mediatypeid']
    except Exception as e:
        self._module.fail_json(msg="Failed to get mediatype '%s': %s" % (mediatype_name, e))