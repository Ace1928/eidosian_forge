import enum
def solve_all_instances(solver_manager, solver, instances, **kwds):
    """
    A simple utility to apply a solver to a list of problem instances.
    """
    solver_manager.solve_all(solver, instances, **kwds)