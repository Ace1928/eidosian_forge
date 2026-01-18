from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.locale import get_best_parsable_locale
from ansible.module_utils.common.respawn import has_respawned, respawn_module
from ansible.module_utils.common.text.converters import to_native, to_text
from ansible.module_utils.yumdnf import YumDnf, yumdnf_argument_spec
import errno
import os
import re
import sys
import tempfile
from contextlib import contextmanager
from ansible.module_utils.urls import fetch_file
@contextmanager
def set_env_proxy(self):
    namepass = ''
    scheme = ['http', 'https']
    old_proxy_env = [os.getenv('http_proxy'), os.getenv('https_proxy')]
    try:
        if self.yum_base.conf.proxy and self.yum_base.conf.proxy not in ('_none_',):
            if self.yum_base.conf.proxy_username:
                namepass = namepass + self.yum_base.conf.proxy_username
                proxy_url = self.yum_base.conf.proxy
                if self.yum_base.conf.proxy_password:
                    namepass = namepass + ':' + self.yum_base.conf.proxy_password
            elif '@' in self.yum_base.conf.proxy:
                namepass = self.yum_base.conf.proxy.split('@')[0].split('//')[-1]
                proxy_url = self.yum_base.conf.proxy.replace('{0}@'.format(namepass), '')
            if namepass:
                namepass = namepass + '@'
                for item in scheme:
                    os.environ[item + '_proxy'] = re.sub('(http://)', '\\g<1>' + namepass, proxy_url)
            else:
                for item in scheme:
                    os.environ[item + '_proxy'] = self.yum_base.conf.proxy
        yield
    except yum.Errors.YumBaseError:
        raise
    finally:
        for item in scheme:
            if os.getenv('{0}_proxy'.format(item)):
                del os.environ['{0}_proxy'.format(item)]
        if old_proxy_env[0]:
            os.environ['http_proxy'] = old_proxy_env[0]
        if old_proxy_env[1]:
            os.environ['https_proxy'] = old_proxy_env[1]