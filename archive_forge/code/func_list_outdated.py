from __future__ import absolute_import, division, print_function
import json
import os
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
def list_outdated(self):
    if not os.path.isfile(os.path.join(self.path, 'pnpm-lock.yaml')):
        return list()
    cmd = ['outdated', '--format', 'json']
    try:
        out, err = self._exec(cmd, True, False)
        if err is not None and err != '':
            raise Exception(out)
        data_lines = out.splitlines(True)
        out = None
        for line in data_lines:
            if len(line) > 0 and line[0] == '{':
                out = line
                continue
            if len(line) > 0 and line[0] == '}':
                out += line
                break
            if out is not None:
                out += line
        data = json.loads(out)
    except Exception as e:
        self.module.fail_json(msg='Failed to parse pnpm output with error %s' % to_native(e))
    return data.keys()