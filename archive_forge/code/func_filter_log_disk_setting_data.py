from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.connection import Connection
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortimanager.common import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.data_post_processor import (
def filter_log_disk_setting_data(json):
    option_list = ['diskfull', 'dlp_archive_quota', 'full_final_warning_threshold', 'full_first_warning_threshold', 'full_second_warning_threshold', 'interface', 'interface_select_method', 'ips_archive', 'log_quota', 'max_log_file_size', 'max_policy_packet_capture_size', 'maximum_log_age', 'report_quota', 'roll_day', 'roll_schedule', 'roll_time', 'source_ip', 'status', 'upload', 'upload_delete_files', 'upload_destination', 'upload_ssl_conn', 'uploaddir', 'uploadip', 'uploadpass', 'uploadport', 'uploadsched', 'uploadtime', 'uploadtype', 'uploaduser']
    json = remove_invalid_fields(json)
    dictionary = {}
    for attribute in option_list:
        if attribute in json and json[attribute] is not None:
            dictionary[attribute] = json[attribute]
    return dictionary