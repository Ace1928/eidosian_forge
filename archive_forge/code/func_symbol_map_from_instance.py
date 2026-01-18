from pyomo.core.expr.symbol_map import SymbolMap
from pyomo.core.base.label import TextLabeler
def symbol_map_from_instance(instance):
    """
    Create a symbol map from an instance using name-based labelers.
    """
    from pyomo.core.base import Var, Constraint, Objective
    symbol_map = SymbolMap()
    labeler = TextLabeler()
    for varvalue in instance.component_data_objects(Var, active=True):
        symbol_map.getSymbol(varvalue, labeler)
    for constraint_data in instance.component_data_objects(Constraint, active=True):
        con_symbol = symbol_map.getSymbol(constraint_data, labeler)
        if constraint_data.equality:
            label = 'c_e_%s_' % con_symbol
            symbol_map.alias(constraint_data, label)
        elif constraint_data.lower is not None:
            if constraint_data.upper is not None:
                symbol_map.alias(constraint_data, 'r_l_%s_' % con_symbol)
                symbol_map.alias(constraint_data, 'r_u_%s_' % con_symbol)
            else:
                label = 'c_l_%s_' % con_symbol
                symbol_map.alias(constraint_data, label)
        elif constraint_data.upper is not None:
            label = 'c_u_%s_' % con_symbol
            symbol_map.alias(constraint_data, label)
    first = True
    for objective_data in instance.component_data_objects(Objective, active=True):
        symbol_map.getSymbol(objective_data, labeler)
        if first:
            symbol_map.alias(objective_data, '__default_objective__')
            first = False
    return symbol_map