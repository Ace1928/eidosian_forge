import pyomo.common.unittest as unittest
import pyomo.kernel as pmo
from pyomo.core.kernel.block import IBlock
from pyomo.core.kernel.variable import IVariable
from pyomo.core.kernel.constraint import IConstraint
def test_component_objects_hack(self):
    model = _model.clone()
    objs = {key: [] for key in [None, IVariable, IConstraint, IBlock, IJunk]}
    for item in pmo.heterogeneous_containers(model):
        objs[None].extend(item.component_objects(descend_into=False))
        self.assertEqual([str(obj) for obj in item.component_objects(descend_into=False)], [str(obj) for obj in item.children()])
        objs[IVariable].extend(item.component_objects(ctype=IVariable, descend_into=False))
        self.assertEqual([str(obj) for obj in item.component_objects(ctype=IVariable, descend_into=False)], [str(obj) for obj in item.children(ctype=IVariable)])
        objs[IConstraint].extend(item.component_objects(ctype=IConstraint, descend_into=False))
        self.assertEqual([str(obj) for obj in item.component_objects(ctype=IConstraint, descend_into=False)], [str(obj) for obj in item.children(ctype=IConstraint)])
        objs[IBlock].extend(item.component_objects(ctype=IBlock, descend_into=False))
        self.assertEqual([str(obj) for obj in item.component_objects(ctype=IBlock, descend_into=False)], [str(obj) for obj in item.children(ctype=IBlock)])
        objs[IJunk].extend(item.component_objects(ctype=IJunk, descend_into=False))
        self.assertEqual([str(obj) for obj in item.component_objects(ctype=IJunk, descend_into=False)], [str(obj) for obj in item.children(ctype=IJunk)])
    all_ = []
    for key in objs:
        if key is None:
            continue
        names = [str(obj) for obj in objs[key]]
        self.assertEqual(sorted([str(obj) for obj in model.component_objects(ctype=key)]), sorted(names))
        all_.extend(names)
    self.assertEqual(sorted([str(obj) for obj in model.component_objects()]), sorted(all_))
    self.assertEqual(sorted([str(obj) for obj in objs[None]]), sorted(all_))
    model.deactivate()
    self.assertEqual(sorted([str(obj) for obj in model.component_objects()]), sorted(all_))
    self.assertEqual([str(obj) for obj in model.component_objects(descend_into=False, active=True)], [])
    self.assertEqual([str(obj) for obj in model.component_objects(descend_into=True, active=True)], [])