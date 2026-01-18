import re
import os_traits as ot
from os_traits.hw.cpu import x86
from os_traits.hw.gpu import api
from os_traits.hw.gpu import resolution
from os_traits.hw.nic import offload
from os_traits.tests import base
def test_get_traits_filter_by_prefix(self):
    traits = ot.get_traits('HW_CPU')
    self.assertIn('HW_CPU_X86_SSE42', traits)
    self.assertIn('HW_CPU_HYPERTHREADING', traits)
    self.assertIn(ot.HW_CPU_X86_AVX2, traits)
    self.assertNotIn(ot.STORAGE_DISK_SSD, traits)
    self.assertNotIn(ot.HW_NIC_SRIOV, traits)
    self.assertNotIn('CUSTOM_NAMESPACE', traits)
    self.assertNotIn('os_traits', traits)