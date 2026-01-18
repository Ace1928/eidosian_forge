from __future__ import absolute_import, division, print_function
import json
import os
import random
import string
import gzip
from io import BytesIO
from ansible.module_utils.urls import open_url
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.common.text.converters import to_text
from ansible.module_utils.common.text.converters import to_bytes
from ansible.module_utils.six import text_type
from ansible.module_utils.six.moves import http_client
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.six.moves.urllib.parse import urlparse
from ansible.module_utils.ansible_release import __version__ as ansible_version
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
def set_boot_override(self, boot_opts):
    bootdevice = boot_opts.get('bootdevice')
    uefi_target = boot_opts.get('uefi_target')
    boot_next = boot_opts.get('boot_next')
    override_enabled = boot_opts.get('override_enabled')
    boot_override_mode = boot_opts.get('boot_override_mode')
    if not bootdevice and override_enabled != 'Disabled':
        return {'ret': False, 'msg': 'bootdevice option required for temporary boot override'}
    response = self.get_request(self.root_uri + self.systems_uri)
    if response['ret'] is False:
        return response
    data = response['data']
    boot = data.get('Boot')
    if boot is None:
        return {'ret': False, 'msg': 'Boot property not found'}
    cur_override_mode = boot.get('BootSourceOverrideMode')
    if override_enabled != 'Disabled':
        annotation = 'BootSourceOverrideTarget@Redfish.AllowableValues'
        if annotation in boot:
            allowable_values = boot[annotation]
            if isinstance(allowable_values, list) and bootdevice not in allowable_values:
                return {'ret': False, 'msg': 'Boot device %s not in list of allowable values (%s)' % (bootdevice, allowable_values)}
    if override_enabled == 'Disabled':
        payload = {'Boot': {'BootSourceOverrideEnabled': override_enabled, 'BootSourceOverrideTarget': 'None'}}
    elif bootdevice == 'UefiTarget':
        if not uefi_target:
            return {'ret': False, 'msg': 'uefi_target option required to SetOneTimeBoot for UefiTarget'}
        payload = {'Boot': {'BootSourceOverrideEnabled': override_enabled, 'BootSourceOverrideTarget': bootdevice, 'UefiTargetBootSourceOverride': uefi_target}}
        if cur_override_mode == 'Legacy':
            payload['Boot']['BootSourceOverrideMode'] = 'UEFI'
    elif bootdevice == 'UefiBootNext':
        if not boot_next:
            return {'ret': False, 'msg': 'boot_next option required to SetOneTimeBoot for UefiBootNext'}
        payload = {'Boot': {'BootSourceOverrideEnabled': override_enabled, 'BootSourceOverrideTarget': bootdevice, 'BootNext': boot_next}}
        if cur_override_mode == 'Legacy':
            payload['Boot']['BootSourceOverrideMode'] = 'UEFI'
    else:
        payload = {'Boot': {'BootSourceOverrideEnabled': override_enabled, 'BootSourceOverrideTarget': bootdevice}}
        if boot_override_mode:
            payload['Boot']['BootSourceOverrideMode'] = boot_override_mode
    resp = self.patch_request(self.root_uri + self.systems_uri, payload, check_pyld=True)
    if resp['ret'] is False:
        vendor = self._get_vendor()['Vendor']
        if vendor == 'Dell':
            if bootdevice == 'UefiTarget' and override_enabled != 'Disabled':
                payload['Boot'].pop('BootSourceOverrideEnabled', None)
                resp = self.patch_request(self.root_uri + self.systems_uri, payload, check_pyld=True)
    if resp['ret'] and resp['changed']:
        resp['msg'] = 'Updated the boot override settings'
    return resp