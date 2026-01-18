from __future__ import absolute_import, division, print_function
import os
import tempfile
import re
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves.urllib.parse import urlparse
from ansible.module_utils.six.moves.urllib.request import getproxies
def import_cert_path(module, executable, path, keystore_path, keystore_pass, alias, keystore_type, trust_cacert):
    """ Import certificate from path into keystore located on
        keystore_path as alias """
    import_cmd = [executable, '-importcert', '-noprompt', '-keystore', keystore_path, '-file', path, '-alias', alias]
    import_cmd += _get_keystore_type_keytool_parameters(keystore_type)
    if trust_cacert:
        import_cmd.extend(['-trustcacerts'])
    import_rc, import_out, import_err = module.run_command(import_cmd, data='%s\n%s' % (keystore_pass, keystore_pass), check_rc=False)
    diff = {'before': '\n', 'after': '%s\n' % alias}
    if import_rc != 0:
        module.fail_json(msg=import_out, rc=import_rc, cmd=import_cmd, error=import_err)
    return dict(changed=True, msg=import_out, rc=import_rc, cmd=import_cmd, stdout=import_out, error=import_err, diff=diff)