from __future__ import absolute_import, division, print_function
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.argspec.system.system import SystemArgs
def system_fact(self):
    fos = self._fos
    vdom = self._module.params['vdom']
    return fos.monitor('system', self._subset['fact'][len('system_'):].replace('_', '/'), vdom=vdom)