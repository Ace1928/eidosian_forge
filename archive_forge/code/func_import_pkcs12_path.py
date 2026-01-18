from __future__ import absolute_import, division, print_function
import os
import tempfile
import re
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves.urllib.parse import urlparse
from ansible.module_utils.six.moves.urllib.request import getproxies
def import_pkcs12_path(module, executable, pkcs12_path, pkcs12_pass, pkcs12_alias, keystore_path, keystore_pass, keystore_alias, keystore_type):
    """ Import pkcs12 from path into keystore located on
        keystore_path as alias """
    import_cmd = [executable, '-importkeystore', '-noprompt', '-srcstoretype', 'pkcs12', '-srckeystore', pkcs12_path, '-srcalias', pkcs12_alias, '-destkeystore', keystore_path, '-destalias', keystore_alias]
    import_cmd += _get_keystore_type_keytool_parameters(keystore_type)
    secret_data = '%s\n%s' % (keystore_pass, pkcs12_pass)
    if not os.path.exists(keystore_path):
        secret_data = '%s\n%s' % (keystore_pass, secret_data)
    import_rc, import_out, import_err = module.run_command(import_cmd, data=secret_data, check_rc=False)
    diff = {'before': '\n', 'after': '%s\n' % keystore_alias}
    if import_rc != 0 or not os.path.exists(keystore_path):
        module.fail_json(msg=import_out, rc=import_rc, cmd=import_cmd, error=import_err)
    return dict(changed=True, msg=import_out, rc=import_rc, cmd=import_cmd, stdout=import_out, error=import_err, diff=diff)