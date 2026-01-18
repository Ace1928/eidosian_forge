from pyomo.core.base.constraint import Constraint
from pyomo.core.base.set import Set
def get_piecewise_constant_constraints(inputs, time, sample_points, use_next=True):
    """Returns an IndexedConstraint that constrains the provided variables
    to be constant between the provided sample points

    Arguments
    ---------
    inputs: list of variables
        Time-indexed variables that will be constrained piecewise constant
    time: Set
        Set of points at which provided variables will be constrained
    sample_points: List of floats
        Points at which "constant constraints" will be omitted; these are
        points at which the provided variables may vary.
    use_next: Bool (default True)
        Whether the next time point will be used in the constant constraint
        at each point in time. Otherwise, the previous time point is used.

    Returns
    -------
    Set, IndexedConstraint
        A RangeSet indexing the list of variables provided and a Constraint
        indexed by the product of this RangeSet and time.

    """
    input_set = Set(initialize=range(len(inputs)))
    sample_point_set = set(sample_points)

    def piecewise_constant_rule(m, i, t):
        if t in sample_point_set:
            return Constraint.Skip
        else:
            var = inputs[i]
            if use_next:
                t_next = time.next(t)
                return var[t] - var[t_next] == 0
            else:
                t_prev = time.prev(t)
                return var[t_prev] - var[t] == 0
    pwc_con = Constraint(input_set, time, rule=piecewise_constant_rule)
    return (input_set, pwc_con)