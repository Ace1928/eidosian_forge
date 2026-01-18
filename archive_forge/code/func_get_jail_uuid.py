from __future__ import (absolute_import, division, print_function)
import subprocess
from ansible_collections.community.general.plugins.connection.jail import Connection as Jail
from ansible.module_utils.common.text.converters import to_native
from ansible.errors import AnsibleError
from ansible.utils.display import Display
def get_jail_uuid(self):
    p = subprocess.Popen([self.iocage_cmd, 'get', 'host_hostuuid', self.ioc_jail], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    stdout, stderr = p.communicate()
    if stdout is not None:
        stdout = to_native(stdout)
    if stderr is not None:
        stderr = to_native(stderr)
    p.wait()
    if p.returncode != 0:
        raise AnsibleError(u'iocage returned an error: {0}'.format(stdout))
    return stdout.strip('\n')