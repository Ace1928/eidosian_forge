from typing import List, Union
import cvxpy.atoms as atoms
from cvxpy.problems.objective import Maximize, Minimize
from cvxpy.transforms import indicator
def targets_and_priorities(objectives: List[Union[Minimize, Maximize]], priorities, targets, limits=None, off_target: float=1e-05) -> Union[Minimize, Maximize]:
    """
    Combines objectives with penalties within a range between target and limit.

    For nonnegative priorities, each Minimize objective i has value

        off_target*objectives[i] when objectives[i] < targets[i]

        (priorities[i]-off_target)*objectives[i] when targets[i] <= objectives[i] <= limits[i]

        +infinity when objectives[i] > limits[i]

    and each Maximize objective i has value

        off_target*objectives[i] when objectives[i] > targets[i]

        (priorities[i]-off_target)*objectives[i] when targets[i] >= objectives[i] >= limits[i]

        -infinity when objectives[i] < limits[i]

    A negative priority flips the objective sense, i.e., we 
    use -objectives[i], -targets[i], and -limits[i] with abs(priorities[i]).

    Args:
      objectives: A list of Minimize/Maximize objectives.
      priorities: The weight within the trange.
      targets: The start (end) of penalty for Minimize (Maximize)
      limits: Optional hard end (start) of penalty for Minimize (Maximize)
      off_target: Penalty outside of target.

    Returns:
      A Minimize/Maximize objective.

    Raises:
      ValueError: If the scalarized objective is neither convex nor concave.
    """
    assert len(objectives) == len(priorities), 'Number of objectives and priorities must match.'
    assert len(objectives) == len(targets), 'Number of objectives and targets must match.'
    if limits is not None:
        assert len(objectives) == len(limits), 'Number of objectives and limits must match.'
    assert off_target >= 0, 'The off_target argument must be nonnegative.'
    num_objs = len(objectives)
    new_objs: List[Union[Minimize, Maximize]] = []
    for i in range(num_objs):
        obj, tar, lim = (objectives[i], targets[i], limits[i] if limits is not None else None)
        if priorities[i] < 0:
            obj, tar, lim = (-obj, -tar, -lim if lim is not None else None)
        sign = 1 if obj.args[0].is_convex() else -1
        delta = sign * (obj.args[0] - targets[i])
        expr = sign * (abs(priorities[i]) - off_target) * atoms.pos(delta)
        expr += off_target * obj.args[0]
        if limits is not None:
            expr += sign * indicator([sign * obj.args[0] <= sign * limits[i]])
        new_objs.append(expr)
    obj_expr = sum(new_objs)
    if obj_expr.is_convex():
        return Minimize(obj_expr)
    elif obj_expr.is_concave():
        return Maximize(obj_expr)
    else:
        raise ValueError('Scalarized objective is neither convex nor concave.')