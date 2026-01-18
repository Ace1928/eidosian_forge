from __future__ import absolute_import, division, print_function
import inspect
import uuid
from ansible.module_utils.compat.paramiko import paramiko
from ansible_collections.ibm.storage_virtualize.plugins.module_utils.ibm_svc_utils import get_logger
def register_plugin(self):
    try:
        cmd = 'svctask registerplugin'
        cmdopts = {}
        name = 'Ansible'
        unique_key = self.username + '_' + str(uuid.getnode())
        caller_class = inspect.stack()[2].frame.f_locals.get('self', None)
        caller_class_name = caller_class.__class__.__name__
        module_name = str(inspect.stack()[3].filename).rsplit('/', maxsplit=1)[-1]
        metadata = module_name[:-3] + ' module with class ' + str(caller_class_name) + ' has been executed by ' + self.username
        cmdopts['name'] = name
        cmdopts['uniquekey'] = unique_key
        cmdopts['version'] = COLLECTION_VERSION
        cmdopts['metadata'] = metadata
        for cmdoptions in cmdopts:
            cmd = cmd + ' -' + cmdoptions + " '" + cmdopts[cmdoptions] + "'"
        self.client.exec_command(cmd)
        return True
    except Exception as e:
        return False