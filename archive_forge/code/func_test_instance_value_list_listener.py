import unittest
from traits.api import (
def test_instance_value_list_listener(self):
    inst = InstanceValueListListener(tc=self)
    inst.trait_set(exp_object=inst.ref, exp_name='value', dst_name='value', exp_old=[0, 1, 2], exp_new=[0, 1, 2, 3], dst_new=[0, 1, 2, 3])
    inst.ref.value = [0, 1, 2, 3]
    self.assertEqual(inst.calls, {x: 1 for x in range(5)})
    self.assertEqual(inst.ref.value, [0, 1, 2, 3])
    inst.reset_traits(['calls'])
    ref = ArgCheckList()
    inst.trait_set(exp_object=inst, exp_name='ref', dst_name='value', exp_old=inst.ref, exp_new=ref, dst_new=[0, 1, 2])
    inst.ref = ref
    self.assertEqual(inst.calls, {x: 1 for x in range(5)})
    self.assertEqual(inst.ref.value, [0, 1, 2])
    inst.reset_traits(['calls'])
    inst.trait_set(exp_object=inst.ref, exp_name='value_items', dst_name='value_items', exp_old=[], exp_new=[3], dst_new=[3])
    inst.ref.value.append(3)
    self.assertEqual(inst.calls, {x: 1 for x in range(5)})
    self.assertEqual(inst.ref.value, [0, 1, 2, 3])
    inst.reset_traits(['calls'])
    inst.trait_set(exp_object=inst.ref, exp_name='value_items', dst_name='value_items', exp_old=[2], exp_new=[], dst_new=[])
    inst.ref.value.pop(2)
    self.assertEqual(inst.calls, {x: 1 for x in range(5)})
    self.assertEqual(inst.ref.value, [0, 1, 3])
    inst.reset_traits(['calls'])
    inst.trait_set(exp_object=inst.ref, exp_name='value_items', dst_name='value_items', exp_old=[1], exp_new=[1, 2], dst_new=[1, 2])
    inst.ref.value[1:2] = [1, 2]
    self.assertEqual(inst.calls, {x: 1 for x in range(5)})
    self.assertEqual(inst.ref.value, [0, 1, 2, 3])