from __future__ import (absolute_import, division, print_function)
import re
from ansible.module_utils.facts.hardware.base import Hardware, HardwareCollector
from ansible.module_utils.facts.utils import get_mount_size
def get_vgs_facts(self):
    """
        Get vg and pv Facts
        rootvg:
        PV_NAME           PV STATE          TOTAL PPs   FREE PPs    FREE DISTRIBUTION
        hdisk0            active            546         0           00..00..00..00..00
        hdisk1            active            546         113         00..00..00..21..92
        realsyncvg:
        PV_NAME           PV STATE          TOTAL PPs   FREE PPs    FREE DISTRIBUTION
        hdisk74           active            1999        6           00..00..00..00..06
        testvg:
        PV_NAME           PV STATE          TOTAL PPs   FREE PPs    FREE DISTRIBUTION
        hdisk105          active            999         838         200..39..199..200..200
        hdisk106          active            999         599         200..00..00..199..200
        """
    vgs_facts = {}
    lsvg_path = self.module.get_bin_path('lsvg')
    xargs_path = self.module.get_bin_path('xargs')
    cmd = '%s -o | %s %s -p' % (lsvg_path, xargs_path, lsvg_path)
    if lsvg_path and xargs_path:
        rc, out, err = self.module.run_command(cmd, use_unsafe_shell=True)
        if rc == 0 and out:
            vgs_facts['vgs'] = {}
            for m in re.finditer('(\\S+):\\n.*FREE DISTRIBUTION(\\n(\\S+)\\s+(\\w+)\\s+(\\d+)\\s+(\\d+).*)+', out):
                vgs_facts['vgs'][m.group(1)] = []
                pp_size = 0
                cmd = '%s %s' % (lsvg_path, m.group(1))
                rc, out, err = self.module.run_command(cmd)
                if rc == 0 and out:
                    pp_size = re.search('PP SIZE:\\s+(\\d+\\s+\\S+)', out).group(1)
                    for n in re.finditer('(\\S+)\\s+(\\w+)\\s+(\\d+)\\s+(\\d+).*', m.group(0)):
                        pv_info = {'pv_name': n.group(1), 'pv_state': n.group(2), 'total_pps': n.group(3), 'free_pps': n.group(4), 'pp_size': pp_size}
                        vgs_facts['vgs'][m.group(1)].append(pv_info)
    return vgs_facts