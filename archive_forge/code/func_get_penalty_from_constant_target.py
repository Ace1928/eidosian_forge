from pyomo.common.collections import ComponentMap
from pyomo.core.base.componentuid import ComponentUID
from pyomo.core.base.expression import Expression
from pyomo.core.base.set import Set
from pyomo.contrib.mpc.data.series_data import get_indexed_cuid
from pyomo.contrib.mpc.data.scalar_data import ScalarData
from pyomo.contrib.mpc.data.series_data import TimeSeriesData
from pyomo.contrib.mpc.data.interval_data import IntervalData
from pyomo.contrib.mpc.data.convert import interval_to_series, _process_to_dynamic_data
def get_penalty_from_constant_target(variables, time, setpoint_data, weight_data=None, variable_set=None):
    """
    This function returns a tracking cost IndexedExpression for the given
    time-indexed variables and associated setpoint data.

    Arguments
    ---------
    variables: list
        List of time-indexed variables to include in the tracking cost
        expression
    time: iterable
        Set of variable indices for which a cost expression will be
        created
    setpoint_data: ScalarData, dict, or ComponentMap
        Maps variable names to setpoint values
    weight_data: ScalarData, dict, or ComponentMap
        Optional. Maps variable names to tracking cost weights. If not
        provided, weights of one are used.
    variable_set: Set
        Optional. A set of indices into the provided list of variables
        by which the cost expression will be indexed.

    Returns
    -------
    Set, Expression
        RangeSet that indexes the list of variables provided and an Expression
        indexed by the RangeSet and time containing the cost term for each
        variable at each point in time.

    """
    if weight_data is None:
        weight_data = ScalarData(ComponentMap(((var, 1.0) for var in variables)))
    if not isinstance(weight_data, ScalarData):
        weight_data = ScalarData(weight_data)
    if not isinstance(setpoint_data, ScalarData):
        setpoint_data = ScalarData(setpoint_data)
    if variable_set is None:
        variable_set = Set(initialize=range(len(variables)))
    for var in variables:
        if not setpoint_data.contains_key(var):
            raise KeyError('Setpoint data dictionary does not contain a key for variable %s' % var.name)
        if not weight_data.contains_key(var):
            raise KeyError('Tracking weight dictionary does not contain a key for variable %s' % var.name)
    cuids = [get_indexed_cuid(var) for var in variables]
    setpoint_data = setpoint_data.get_data()
    weight_data = weight_data.get_data()

    def tracking_rule(m, i, t):
        return get_quadratic_penalty_at_time(variables[i], t, setpoint_data[cuids[i]], weight=weight_data[cuids[i]])
    tracking_expr = Expression(variable_set, time, rule=tracking_rule)
    return (variable_set, tracking_expr)