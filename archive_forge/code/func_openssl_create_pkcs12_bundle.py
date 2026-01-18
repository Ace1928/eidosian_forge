from __future__ import (absolute_import, division, print_function)
import os
import re
import tempfile
from ansible.module_utils.six import PY2
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.common.text.converters import to_bytes, to_native
def openssl_create_pkcs12_bundle(self, keystore_p12_path):
    export_p12_cmd = [self.openssl_bin, 'pkcs12', '-export', '-name', self.name, '-in', self.certificate_path, '-inkey', self.private_key_path, '-out', keystore_p12_path, '-passout', 'stdin']
    cmd_stdin = ''
    if self.keypass:
        export_p12_cmd.append('-passin')
        export_p12_cmd.append('stdin')
        cmd_stdin = '%s\n' % self.keypass
    cmd_stdin += '%s\n%s' % (self.password, self.password)
    rc, export_p12_out, export_p12_err = self.module.run_command(export_p12_cmd, data=cmd_stdin, environ_update=None, check_rc=False)
    self.result = dict(msg=export_p12_out, cmd=export_p12_cmd, rc=rc)
    if rc != 0:
        self.result['err'] = export_p12_err
        self.module.fail_json(**self.result)