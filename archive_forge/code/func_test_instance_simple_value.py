import unittest
from traits.api import (
def test_instance_simple_value(self):
    inst = InstanceSimpleValue(tc=self)
    for i in range(3):
        inst.trait_set(exp_object=inst.ref, exp_name='value', dst_name='value', exp_old=i, exp_new=i + 1, dst_new=i + 1)
        inst.ref.value = i + 1
    self.assertEqual(inst.calls, {x: 3 for x in range(5)})
    self.assertEqual(inst.ref.value, 3)
    inst.reset_traits(['calls'])
    ref = ArgCheckBase()
    inst.trait_set(exp_object=inst, exp_name='ref', dst_name='value', exp_old=inst.ref, exp_new=ref, dst_new=0)
    inst.ref = ref
    self.assertEqual(inst.calls, {x: 1 for x in range(5)})
    self.assertEqual(inst.ref.value, 0)
    inst.reset_traits(['calls'])
    for i in range(3):
        inst.trait_set(exp_object=inst.ref, exp_name='value', dst_name='value', exp_old=i, exp_new=i + 1, dst_new=i + 1)
        inst.ref.value = i + 1
    self.assertEqual(inst.calls, {x: 3 for x in range(5)})
    self.assertEqual(inst.ref.value, 3)