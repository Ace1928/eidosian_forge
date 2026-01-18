from __future__ import (absolute_import, division, print_function)
import re
import time
from ansible.module_utils.common.locale import get_best_parsable_locale
from ansible.module_utils.common.text.formatters import bytes_to_human
from ansible.module_utils.facts.utils import get_file_content, get_mount_size
from ansible.module_utils.facts.hardware.base import Hardware, HardwareCollector
from ansible.module_utils.facts import timeout
from ansible.module_utils.six.moves import reduce
def get_dmi_facts(self):
    dmi_facts = {}
    rc, platform, err = self.module.run_command('/usr/bin/uname -i')
    platform_sbin = '/usr/platform/' + platform.rstrip() + '/sbin'
    prtdiag_path = self.module.get_bin_path('prtdiag', opt_dirs=[platform_sbin])
    rc, out, err = self.module.run_command(prtdiag_path)
    if out:
        system_conf = out.split('\n')[0]
        vendors = ['Fujitsu', 'Oracle Corporation', 'QEMU', 'Sun Microsystems', 'VMware, Inc.']
        vendor_regexp = '|'.join(map(re.escape, vendors))
        system_conf_regexp = 'System Configuration:\\s+' + '(' + vendor_regexp + ')\\s+' + '(?:sun\\w+\\s+)?' + '(.+)'
        found = re.match(system_conf_regexp, system_conf)
        if found:
            dmi_facts['system_vendor'] = found.group(1)
            dmi_facts['product_name'] = found.group(2)
    return dmi_facts