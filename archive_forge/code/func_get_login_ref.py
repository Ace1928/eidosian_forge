from __future__ import absolute_import, division, print_function
import os
from .common import (
from .constants import (
from .icontrol import iControlRestSession
def get_login_ref(self, provider):
    info = self.read_provider_info_from_device()
    uuids = [os.path.basename(os.path.dirname(x['link'])) for x in info['providers'] if '-' in x['link']]
    if provider in uuids:
        link = self._get_login_ref_by_id(info, provider)
        if not link:
            raise F5ModuleError('Provider with the UUID {0} was not found.'.format(provider))
        return dict(loginReference=dict(link=link))
    names = [os.path.basename(os.path.dirname(x['link'])) for x in info['providers'] if '-' in x['link']]
    if names.count(provider) > 1:
        raise F5ModuleError('Ambiguous auth_provider name provided. Please specify a specific provider name or UUID.')
    link = self._get_login_ref_by_name(info, provider)
    if not link:
        raise F5ModuleError("Provider with the name '{0}' was not found.".format(provider))
    return dict(loginReference=dict(link=link))