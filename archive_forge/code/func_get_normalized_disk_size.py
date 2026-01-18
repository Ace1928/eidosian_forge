from __future__ import (absolute_import, division, print_function)
import re
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.network import is_mac
from ansible.module_utils import six
from ansible_collections.community.general.plugins.module_utils.xenserver import (
def get_normalized_disk_size(self, disk_params, msg_prefix=''):
    """Parses disk size parameters and returns disk size in bytes.

        This method tries to parse disk size module parameters. It fails
        with an error message if size cannot be parsed.

        Args:
            disk_params (dist): A dictionary with disk parameters.
            msg_prefix (str): A string error messages should be prefixed
                with (default: "").

        Returns:
            int: disk size in bytes if disk size is successfully parsed or
            None if no disk size parameters were found.
        """
    disk_size_spec = [x for x in disk_params.keys() if disk_params[x] is not None and (x.startswith('size_') or x == 'size')]
    if disk_size_spec:
        try:
            if 'size' in disk_size_spec:
                size_regex = re.compile('(\\d+(?:\\.\\d+)?)\\s*(.*)')
                disk_size_m = size_regex.match(disk_params['size'])
                if disk_size_m:
                    size = disk_size_m.group(1)
                    unit = disk_size_m.group(2)
                else:
                    raise ValueError
            else:
                size = disk_params[disk_size_spec[0]]
                unit = disk_size_spec[0].split('_')[-1]
            if not unit:
                unit = 'b'
            else:
                unit = unit.lower()
            if re.match('\\d+\\.\\d+', size):
                if unit == 'b':
                    size = int(float(size))
                else:
                    size = float(size)
            else:
                size = int(size)
            if not size or size < 0:
                raise ValueError
        except (TypeError, ValueError, NameError):
            self.module.fail_json(msg='%sfailed to parse disk size! Please review value provided using documentation.' % msg_prefix)
        disk_units = dict(tb=4, gb=3, mb=2, kb=1, b=0)
        if unit in disk_units:
            return int(size * 1024 ** disk_units[unit])
        else:
            self.module.fail_json(msg="%s'%s' is not a supported unit for disk size! Supported units are ['%s']." % (msg_prefix, unit, "', '".join(sorted(disk_units.keys(), key=lambda key: disk_units[key]))))
    else:
        return None