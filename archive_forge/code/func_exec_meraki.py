from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import env_fallback
from ansible.module_utils._text import to_native
import os.path
def exec_meraki(self, family, function, params=None, op_modifies=False, **kwargs):
    try:
        family = getattr(self.api, family)
        func = getattr(family, function)
    except Exception as e:
        self.fail_json(msg=e)
    try:
        if params:
            file_paths_params = kwargs.get('file_paths', [])
            if file_paths_params and isinstance(file_paths_params, list):
                multipart_fields = {}
                for key, value in file_paths_params:
                    if isinstance(params.get(key), str) and self.is_file(params[key]):
                        file_name = self.extract_file_name(params[key])
                        file_path = params[key]
                        multipart_fields[value] = (file_name, open(file_path, 'rb'))
                params.setdefault('multipart_fields', multipart_fields)
                params.setdefault('multipart_monitor_callback', None)
            response = func(**params)
        else:
            response = func()
    except exceptions.APIError as e:
        self.fail_json(msg='An error occured when executing operation.The error was: {error}'.format(error=to_native(e)))
    return response