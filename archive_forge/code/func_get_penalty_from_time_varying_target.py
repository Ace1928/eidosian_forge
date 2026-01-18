from pyomo.common.collections import ComponentMap
from pyomo.core.base.componentuid import ComponentUID
from pyomo.core.base.expression import Expression
from pyomo.core.base.set import Set
from pyomo.contrib.mpc.data.series_data import get_indexed_cuid
from pyomo.contrib.mpc.data.scalar_data import ScalarData
from pyomo.contrib.mpc.data.series_data import TimeSeriesData
from pyomo.contrib.mpc.data.interval_data import IntervalData
from pyomo.contrib.mpc.data.convert import interval_to_series, _process_to_dynamic_data
def get_penalty_from_time_varying_target(variables, time, setpoint_data, weight_data=None, variable_set=None):
    """Constructs a penalty expression for the specified variables and
    specified time-varying target data.

    Arguments
    ---------
    variables: List of Pyomo variables
        Variables that participate in the cost expressions.
    time: Iterable
        Index used for the cost expression
    setpoint_data: TimeSeriesData
        Holds the trajectory values that will be used as a setpoint
    weight_data: ScalarData (optional)
        Weights for variables. Default is all ones.
    variable_set: Set (optional)
        Set indexing the list of provided variables, if one exists already.

    Returns
    -------
    Set, Expression
        Set indexing the list of provided variables and Expression, indexed
        by the variable set and time, for the total weighted penalty with
        respect to the provided setpoint.

    """
    if variable_set is None:
        variable_set = Set(initialize=range(len(variables)))
    tracking_costs = _get_penalty_expressions_from_time_varying_target(variables, time, setpoint_data, weight_data=weight_data)

    def tracking_rule(m, i, t):
        return tracking_costs[i][t]
    tracking_cost = Expression(variable_set, time, rule=tracking_rule)
    return (variable_set, tracking_cost)