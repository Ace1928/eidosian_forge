from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.connection import Connection
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortimanager.common import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.data_post_processor import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.comparison import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.comparison import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.comparison import (
def filter_system_admin_data(json):
    option_list = ['accprofile', 'accprofile_override', 'allow_remove_admin_session', 'comments', 'email_to', 'force_password_change', 'fortitoken', 'guest_auth', 'guest_lang', 'guest_usergroups', 'gui_dashboard', 'gui_global_menu_favorites', 'gui_new_feature_acknowledge', 'gui_vdom_menu_favorites', 'hidden', 'history0', 'history1', 'ip6_trusthost1', 'ip6_trusthost10', 'ip6_trusthost2', 'ip6_trusthost3', 'ip6_trusthost4', 'ip6_trusthost5', 'ip6_trusthost6', 'ip6_trusthost7', 'ip6_trusthost8', 'ip6_trusthost9', 'login_time', 'name', 'password', 'password_expire', 'peer_auth', 'peer_group', 'radius_vdom_override', 'remote_auth', 'remote_group', 'schedule', 'sms_custom_server', 'sms_phone', 'sms_server', 'ssh_certificate', 'ssh_public_key1', 'ssh_public_key2', 'ssh_public_key3', 'trusthost1', 'trusthost10', 'trusthost2', 'trusthost3', 'trusthost4', 'trusthost5', 'trusthost6', 'trusthost7', 'trusthost8', 'trusthost9', 'two_factor', 'two_factor_authentication', 'two_factor_notification', 'vdom', 'vdom_override', 'wildcard']
    json = remove_invalid_fields(json)
    dictionary = {}
    for attribute in option_list:
        if attribute in json and json[attribute] is not None:
            dictionary[attribute] = json[attribute]
    return dictionary