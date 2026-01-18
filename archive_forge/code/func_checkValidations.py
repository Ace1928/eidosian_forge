from __future__ import absolute_import, division, print_function
from ansible.module_utils._text import to_text
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.connection import ConnectionError, exec_command
from ansible_collections.community.network.plugins.module_utils.network.icx.icx import exec_scp, run_commands
def checkValidations(module):
    validation = dict(scp=dict(upload=['running-config', 'startup-config', 'flash_primary', 'flash_secondary'], download=['running-config', 'startup-config', 'flash_primary', 'flash_secondary', 'bootrom', 'fips-primary-sig', 'fips-secondary-sig', 'fips-bootrom-sig']), https=dict(upload=['running-config', 'startup-config'], download=['flash_primary', 'flash_secondary', 'startup-config']))
    protocol = module.params['protocol']
    upload = module.params['upload']
    download = module.params['download']
    if protocol == 'scp' and module.params['remote_user'] is None:
        module.fail_json(msg='While using scp remote_user argument is required')
    if upload is None and download is None:
        module.fail_json(msg='Upload or download params are required.')
    if upload is not None and download is not None:
        module.fail_json(msg='Only upload or download can be used at a time.')
    if upload:
        if not upload in validation.get(protocol).get('upload'):
            module.fail_json(msg="Specified resource '" + upload + "' can't be uploaded to '" + protocol + "'")
    if download:
        if not download in validation.get(protocol).get('download'):
            module.fail_json(msg="Specified resource '" + download + "' can't be downloaded from '" + protocol + "'")