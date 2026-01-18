import pyomo.common.unittest as unittest
import pyomo.kernel as pmo
from pyomo.core.kernel.block import IBlock
from pyomo.core.kernel.variable import IVariable
from pyomo.core.kernel.constraint import IConstraint
def test_component_data_objects_hack(self):
    model = _model.clone()
    self.assertEqual([str(obj) for obj in model.component_data_objects()], [str(obj) for obj in model.components()])
    self.assertEqual([str(obj) for obj in model.component_data_objects(ctype=IVariable)], [str(obj) for obj in model.components(ctype=IVariable)])
    self.assertEqual([str(obj) for obj in model.component_data_objects(ctype=IConstraint)], [str(obj) for obj in model.components(ctype=IConstraint)])
    self.assertEqual([str(obj) for obj in model.component_data_objects(ctype=IBlock)], [str(obj) for obj in model.components(ctype=IBlock)])
    self.assertEqual([str(obj) for obj in model.component_data_objects(ctype=IJunk)], [str(obj) for obj in model.components(ctype=IJunk)])
    for item in pmo.preorder_traversal(model):
        item.deactivate()
        self.assertEqual([str(obj) for obj in model.component_data_objects(active=True)], [str(obj) for obj in model.components(active=True)])
        self.assertEqual([str(obj) for obj in model.component_data_objects(ctype=IVariable, active=True)], [str(obj) for obj in model.components(ctype=IVariable, active=True)])
        self.assertEqual([str(obj) for obj in model.component_data_objects(ctype=IConstraint, active=True)], [str(obj) for obj in model.components(ctype=IConstraint, active=True)])
        self.assertEqual([str(obj) for obj in model.component_data_objects(ctype=IBlock, active=True)], [str(obj) for obj in model.components(ctype=IBlock, active=True)])
        self.assertEqual([str(obj) for obj in model.component_data_objects(ctype=IJunk, active=True)], [str(obj) for obj in model.components(ctype=IJunk, active=True)])
        item.activate()