from __future__ import (absolute_import, division, print_function)
import os
from tempfile import NamedTemporaryFile
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import open_url
from ansible.module_utils.common.text.converters import to_native
def download_SAPCAR(binary_path, module):
    bin_path = None
    if binary_path is not None:
        if binary_path.startswith('https://') or binary_path.startswith('http://'):
            random_file = NamedTemporaryFile(delete=False)
            with open_url(binary_path) as response:
                with random_file as out_file:
                    data = response.read()
                    out_file.write(data)
            os.chmod(out_file.name, 448)
            bin_path = out_file.name
            module.add_cleanup_file(bin_path)
        else:
            bin_path = binary_path
    return bin_path