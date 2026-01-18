from __future__ import (absolute_import, division, print_function)
import os
from tempfile import NamedTemporaryFile
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import open_url
from ansible.module_utils.common.text.converters import to_native
def check_if_present(command, path, dest, signature, manifest, module):
    iter_command = [command, '-tvf', path]
    sar_out = module.run_command(iter_command)[1]
    sar_raw = sar_out.split('\n')[1:]
    if dest[-1] != '/':
        dest = dest + '/'
    sar_files = [dest + x.split(' ')[-1] for x in sar_raw if x]
    if not signature:
        sar_files = [item for item in sar_files if not item.endswith('.SMF')]
    if manifest != 'SIGNATURE.SMF':
        sar_files = [item for item in sar_files if not item.endswith('.SMF')]
        sar_files = sar_files + [manifest]
    files_extracted = get_list_of_files(dest)
    present = all((elem in files_extracted for elem in sar_files))
    return present