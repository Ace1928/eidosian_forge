import pyomo.common.unittest as unittest
import pyomo.environ as pe
from pyomo.contrib.solver.gurobi import Gurobi
from pyomo.contrib.solver.results import SolutionStatus
from pyomo.core.expr.taylor_series import taylor_series_expansion
import gurobipy
def create_pmedian_model():
    d_dict = {(1, 1): 1.777356642700564, (1, 2): 1.6698255595592497, (1, 3): 1.099139603924817, (1, 4): 1.3529705111901453, (1, 5): 1.467907742900842, (1, 6): 1.5346837414708774, (2, 1): 1.9783090609123972, (2, 2): 1.130315350158659, (2, 3): 1.6712434682302661, (2, 4): 1.3642294159473756, (2, 5): 1.4888357071619858, (2, 6): 1.2030122107340537, (3, 1): 1.6661983755713592, (3, 2): 1.227663031206932, (3, 3): 1.4580640582967632, (3, 4): 1.0407223975549575, (3, 5): 1.9742897953778287, (3, 6): 1.4874760742689066, (4, 1): 1.4616138636373597, (4, 2): 1.7141471558082002, (4, 3): 1.4157281494999725, (4, 4): 1.888011688001529, (4, 5): 1.0232934487237717, (4, 6): 1.8335062677845464, (5, 1): 1.468494740997508, (5, 2): 1.8114798126442795, (5, 3): 1.9455914886158723, (5, 4): 1.983088378194899, (5, 5): 1.1761820755785306, (5, 6): 1.698655759576308, (6, 1): 1.108855711312383, (6, 2): 1.1602637342062019, (6, 3): 1.0928602740245892, (6, 4): 1.3140620798928404, (6, 5): 1.0165386843386672, (6, 6): 1.854049125736362, (7, 1): 1.2910160386456968, (7, 2): 1.7800475863350327, (7, 3): 1.5480965161255695, (7, 4): 1.1943306766997612, (7, 5): 1.2920382721805297, (7, 6): 1.3194527773994338, (8, 1): 1.6585982235379078, (8, 2): 1.2315210354122292, (8, 3): 1.6194303369953538, (8, 4): 1.8953386098022103, (8, 5): 1.8694342085696831, (8, 6): 1.2938069356684523, (9, 1): 1.4582048085805495, (9, 2): 1.484979797871119, (9, 3): 1.2803882693587225, (9, 4): 1.3289569463506004, (9, 5): 1.9842424240265042, (9, 6): 1.0119441379208745, (10, 1): 1.1429007682932852, (10, 2): 1.6519772165446711, (10, 3): 1.0749931799469326, (10, 4): 1.2920787022811089, (10, 5): 1.7934429721917704, (10, 6): 1.9115931008709737}
    model = pe.ConcreteModel()
    model.N = pe.Param(initialize=10)
    model.Locations = pe.RangeSet(1, model.N)
    model.P = pe.Param(initialize=3)
    model.M = pe.Param(initialize=6)
    model.Customers = pe.RangeSet(1, model.M)
    model.d = pe.Param(model.Locations, model.Customers, initialize=d_dict, within=pe.Reals)
    model.x = pe.Var(model.Locations, model.Customers, bounds=(0.0, 1.0))
    model.y = pe.Var(model.Locations, within=pe.Binary)

    def rule(model):
        return sum((model.d[n, m] * model.x[n, m] for n in model.Locations for m in model.Customers))
    model.obj = pe.Objective(rule=rule)

    def rule(model, m):
        return (sum((model.x[n, m] for n in model.Locations)), 1.0)
    model.single_x = pe.Constraint(model.Customers, rule=rule)

    def rule(model, n, m):
        return (None, model.x[n, m] - model.y[n], 0.0)
    model.bound_y = pe.Constraint(model.Locations, model.Customers, rule=rule)

    def rule(model):
        return (sum((model.y[n] for n in model.Locations)) - model.P, 0.0)
    model.num_facilities = pe.Constraint(rule=rule)
    return model