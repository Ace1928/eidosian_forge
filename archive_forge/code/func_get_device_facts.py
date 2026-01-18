from __future__ import (absolute_import, division, print_function)
import re
import time
from ansible.module_utils.common.locale import get_best_parsable_locale
from ansible.module_utils.common.text.formatters import bytes_to_human
from ansible.module_utils.facts.utils import get_file_content, get_mount_size
from ansible.module_utils.facts.hardware.base import Hardware, HardwareCollector
from ansible.module_utils.facts import timeout
from ansible.module_utils.six.moves import reduce
def get_device_facts(self):
    device_facts = {}
    device_facts['devices'] = {}
    disk_stats = {'Product': 'product', 'Revision': 'revision', 'Serial No': 'serial', 'Size': 'size', 'Vendor': 'vendor', 'Hard Errors': 'hard_errors', 'Soft Errors': 'soft_errors', 'Transport Errors': 'transport_errors', 'Media Error': 'media_errors', 'Predictive Failure Analysis': 'predictive_failure_analysis', 'Illegal Request': 'illegal_request'}
    cmd = ['/usr/bin/kstat', '-p']
    for ds in disk_stats:
        cmd.append('sderr:::%s' % ds)
    d = {}
    rc, out, err = self.module.run_command(cmd)
    if rc != 0:
        return device_facts
    sd_instances = frozenset((line.split(':')[1] for line in out.split('\n') if line.startswith('sderr')))
    for instance in sd_instances:
        lines = (line for line in out.split('\n') if ':' in line and line.split(':')[1] == instance)
        for line in lines:
            text, value = line.split('\t')
            stat = text.split(':')[3]
            if stat == 'Size':
                d[disk_stats.get(stat)] = bytes_to_human(float(value))
            else:
                d[disk_stats.get(stat)] = value.rstrip()
        diskname = 'sd' + instance
        device_facts['devices'][diskname] = d
        d = {}
    return device_facts