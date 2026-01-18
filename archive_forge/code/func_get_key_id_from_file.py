from __future__ import absolute_import, division, print_function
import os
from traceback import format_exc
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.locale import get_best_parsable_locale
from ansible.module_utils.urls import fetch_url
def get_key_id_from_file(module, filename, data=None):
    native_data = to_native(data)
    is_armored = native_data.find('-----BEGIN PGP PUBLIC KEY BLOCK-----') >= 0
    key = None
    cmd = [gpg_bin, '--with-colons', filename]
    rc, out, err = module.run_command(cmd, environ_update=lang_env(module), data=native_data if is_armored else data, binary_data=not is_armored)
    if rc != 0:
        module.fail_json(msg="Unable to extract key from '%s'" % ('inline data' if data is not None else filename), stdout=out, stderr=err)
    keys = parse_output_for_keys(out)
    if keys:
        key = keys[0]
    return key