from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_utils import (
from ansible.module_utils._text import to_native
def delete_cloud_backup(self):
    if self.module.check_mode:
        self.changed = True
        return
    cmd = 'rmvolumebackupgeneration'
    cmdopts = {}
    if self.volume_name:
        cmdopts['volume'] = self.volume_name
        var = self.volume_name
        self.msg = 'Cloud backup ({0}) deleted'.format(self.volume_name)
    else:
        cmdopts['uid'] = self.volume_UID
        var = self.volume_UID
        self.msg = 'Cloud backup ({0}) deleted'.format(self.volume_UID)
    if self.generation:
        cmdopts['generation'] = self.generation
    if self.all not in {'', None}:
        cmdopts['all'] = self.all
    response = self.restapi._svc_token_wrap(cmd, cmdopts=cmdopts, cmdargs=None)
    self.log('response=%s', response)
    self.changed = True
    if response['out']:
        if b'CMMVC9104E' in response['out']:
            self.changed = False
            self.msg = 'CMMVC9104E: Volume ({0}) is not ready to perform any operation right now.'.format(var)
        elif b'CMMVC9090E' in response['out']:
            self.changed = False
            self.msg = 'Cloud backup generation already deleted.'
        else:
            self.module.fail_json(msg=response)
    self.log(self.msg)