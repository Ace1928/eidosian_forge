from __future__ import absolute_import, division, print_function
import re
from time import sleep
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.nxos import (
def check_mode_legacy(module, issu, image, kick=None):
    """Some platforms/images/transports don't support the 'install all impact'
    command so we need to use a different method."""
    current = execute_show_command(module, 'show version', 'json')[0]
    data = parse_show_install('')
    upgrade_msg = 'No upgrade required'
    data['error'] = False
    tsver = 'show version image bootflash:%s' % image
    data['upgrade_cmd'] = [tsver]
    target_image = parse_show_version(execute_show_command(module, tsver))
    if target_image['error']:
        data['error'] = True
        data['raw'] = target_image['raw']
    if current['kickstart_ver_str'] != target_image['version'] and (not data['error']):
        data['upgrade_needed'] = True
        data['disruptive'] = True
        upgrade_msg = 'Switch upgraded: system: %s' % tsver
    if kick is not None and (not data['error']):
        tkver = 'show version image bootflash:%s' % kick
        data['upgrade_cmd'].append(tsver)
        target_kick = parse_show_version(execute_show_command(module, tkver))
        if target_kick['error']:
            data['error'] = True
            data['raw'] = target_kick['raw']
        if current['kickstart_ver_str'] != target_kick['version'] and (not data['error']):
            data['upgrade_needed'] = True
            data['disruptive'] = True
            upgrade_msg = upgrade_msg + ' kickstart: %s' % tkver
    data['list_data'] = data['raw']
    data['processed'] = upgrade_msg
    return data