from __future__ import absolute_import, division, print_function
import hashlib
import io
import os
import re
import ssl
import sys
import tarfile
import time
import traceback
import xml.etree.ElementTree as ET
from threading import Thread
from ansible.module_utils._text import to_native
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import string_types
from ansible.module_utils.six.moves.urllib.request import Request, urlopen
from ansible.module_utils.urls import generic_urlparse, open_url, urlparse, urlunparse
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
def inject_ovf_env(self):
    attrib = {'xmlns': 'http://schemas.dmtf.org/ovf/environment/1', 'xmlns:xsi': 'http://www.w3.org/2001/XMLSchema-instance', 'xmlns:oe': 'http://schemas.dmtf.org/ovf/environment/1', 'xmlns:ve': 'http://www.vmware.com/schema/ovfenv', 'oe:id': '', 've:esxId': self.entity._moId}
    env = ET.Element('Environment', **attrib)
    platform = ET.SubElement(env, 'PlatformSection')
    ET.SubElement(platform, 'Kind').text = self.content.about.name
    ET.SubElement(platform, 'Version').text = self.content.about.version
    ET.SubElement(platform, 'Vendor').text = self.content.about.vendor
    ET.SubElement(platform, 'Locale').text = 'US'
    prop_section = ET.SubElement(env, 'PropertySection')
    for key, value in self.params['properties'].items():
        params = {'oe:key': key, 'oe:value': str(value) if isinstance(value, bool) else value}
        ET.SubElement(prop_section, 'Property', **params)
    opt = vim.option.OptionValue()
    opt.key = 'guestinfo.ovfEnv'
    opt.value = '<?xml version="1.0" encoding="UTF-8"?>' + to_native(ET.tostring(env))
    config_spec = vim.vm.ConfigSpec()
    config_spec.extraConfig = [opt]
    task = self.entity.ReconfigVM_Task(config_spec)
    wait_for_task(task)