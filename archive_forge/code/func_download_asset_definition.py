from __future__ import absolute_import, division, print_function
from ansible.module_utils.six import text_type
from ansible.plugins.action import ActionBase
from ansible.utils.vars import merge_hash
from ..module_utils import bonsai, errors
def download_asset_definition(self, on_remote, name, version, task_vars):
    if not on_remote:
        return bonsai.get_asset_parameters(name, version)
    args = dict(name=name, version=version)
    result = self._execute_module(module_name='sensu.sensu_go.bonsai_asset', module_args=args, task_vars=task_vars, wrap_async=False)
    if result.get('failed', False):
        raise errors.Error(result['msg'])
    return result['asset']