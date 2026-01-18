from __future__ import absolute_import, division, print_function
import re
import traceback
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import (
from ..module_utils.teem import send_teem
@property
def prober_preference(self):
    if self.want.prober_preference is None:
        return None
    if self.want.prober_preference == self.have.prober_preference:
        return None
    if self.want.prober_preference == 'pool' and self.want.prober_pool is None:
        raise F5ModuleError("A prober_pool needs to be set if prober_preference is set to 'pool'")
    if self.want.prober_preference != 'pool' and self.have.prober_preference == 'pool':
        if self.want.prober_fallback != 'pool' and self.want.prober_pool != '':
            raise F5ModuleError('To change prober_preference from {0} to {1}, set prober_pool to an empty string'.format(self.have.prober_preference, self.want.prober_preference))
    if self.want.prober_preference == self.want.prober_fallback:
        raise F5ModuleError('Prober_preference and prober_fallback must not be equal.')
    if self.want.prober_preference == self.have.prober_fallback:
        raise F5ModuleError('Cannot set prober_preference to {0} if prober_fallback on device is set to {1}.'.format(self.want.prober_preference, self.have.prober_fallback))
    if self.want.prober_preference != self.have.prober_preference:
        return self.want.prober_preference