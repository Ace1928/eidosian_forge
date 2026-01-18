from pyomo.core.base.componentuid import ComponentUID
from pyomo.core.base.expression import Expression
from pyomo.core.base.set import Set
from pyomo.contrib.mpc.data.series_data import get_indexed_cuid
from pyomo.contrib.mpc.data.scalar_data import ScalarData
def get_terminal_penalty(variables, time_set, target_data, weight_data=None, variable_set=None):
    """Returns an Expression penalizing the deviation of the specified
    variables at the final point in time from the specified target

    Arguments
    ---------
    variables: List
        List of time-indexed variables that will be penalized
    time_set: Set
        Time set that indexes the provided variables. Penalties are applied
        at the last point in this set.
    target_data: ScalarData
        ScalarData object containing the target for (at least) the variables
        to be penalized
    weight_data: ScalarData (optional)
        ScalarData object containing the penalty weights for (at least) the
        variables to be penalized
    variable_set: Set (optional)
        Set indexing the list of variables provided, if such a set already
        exists

    Returns
    -------
    Set, Expression
        Set indexing the list of variables provided and an Expression,
        indexed by this set, containing the weighted penalty expressions

    """
    t = time_set.last()
    return get_penalty_at_time(variables, t, target_data, weight_data=weight_data, time_set=time_set, variable_set=variable_set)