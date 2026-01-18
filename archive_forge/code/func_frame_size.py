from __future__ import absolute_import, division, print_function
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
@property
def frame_size(self):
    frame = self._values['frame_size']
    if frame is None:
        return None
    if frame < 1024 or frame > 16384:
        raise F5ModuleError('Write Size value must be between 1024 and 16384')
    return self._values['frame_size']