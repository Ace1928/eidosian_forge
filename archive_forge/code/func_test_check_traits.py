import re
import os_traits as ot
from os_traits.hw.cpu import x86
from os_traits.hw.gpu import api
from os_traits.hw.gpu import resolution
from os_traits.hw.nic import offload
from os_traits.tests import base
def test_check_traits(self):
    traits = set(['HW_CPU_X86_SSE42', 'HW_CPU_X86_XOP'])
    not_traits = set(['not_trait1', 'not_trait2'])
    check_traits = []
    check_traits.extend(traits)
    check_traits.extend(not_traits)
    self.assertEqual((traits, not_traits), ot.check_traits(check_traits))