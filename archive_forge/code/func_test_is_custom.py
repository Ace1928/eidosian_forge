import re
import os_traits as ot
from os_traits.hw.cpu import x86
from os_traits.hw.gpu import api
from os_traits.hw.gpu import resolution
from os_traits.hw.nic import offload
from os_traits.tests import base
def test_is_custom(self):
    self.assertTrue(ot.is_custom('CUSTOM_FOO'))
    self.assertFalse(ot.is_custom('HW_CPU_X86_SSE42'))