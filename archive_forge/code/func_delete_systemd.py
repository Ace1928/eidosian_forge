from __future__ import absolute_import, division, print_function
import json
import os
import shutil
from ansible.module_utils.six import raise_from
def delete_systemd(module, module_params, name, version):
    sysconf = module_params['generate_systemd']
    if not sysconf.get('path'):
        module.log('PODMAN-CONTAINER-DEBUG: Not deleting systemd file - no path!')
        return
    rc, systemd, err = run_generate_systemd_command(module, module_params, name, version)
    if rc != 0:
        module.log('PODMAN-CONTAINER-DEBUG: Error generating systemd: %s' % err)
        return
    else:
        try:
            data = json.loads(systemd)
            for file_name in data.keys():
                file_name += '.service'
                full_dir_path = os.path.expanduser(sysconf['path'])
                file_path = os.path.join(full_dir_path, file_name)
                if os.path.exists(file_path):
                    os.unlink(file_path)
            return
        except Exception as e:
            module.log('PODMAN-CONTAINER-DEBUG: Error deleting systemd: %s' % e)
            return