from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
def parse_lsusb(module, lsusb_path):
    rc, stdout, stderr = module.run_command(lsusb_path, check_rc=True)
    regex = re.compile('^Bus (\\d{3}) Device (\\d{3}): ID ([0-9a-f]{4}:[0-9a-f]{4}) (.*)$')
    usb_devices = []
    for line in stdout.splitlines():
        match = re.match(regex, line)
        if not match:
            module.fail_json(msg='failed to parse unknown lsusb output %s' % line, stdout=stdout, stderr=stderr)
        current_device = {'bus': match.group(1), 'device': match.group(2), 'id': match.group(3), 'name': match.group(4)}
        usb_devices.append(current_device)
    return_value = {'usb_devices': usb_devices}
    module.exit_json(msg='parsed %s USB devices' % len(usb_devices), stdout=stdout, stderr=stderr, ansible_facts=return_value)