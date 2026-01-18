import enum
from pyomo.common.config import ConfigDict, ConfigValue, InEnum
from pyomo.common.modeling import NOTSET
from pyomo.repn.plugins.nl_writer import AMPLRepnVisitor, text_nl_template
from pyomo.repn.util import FileDeterminism, FileDeterminism_to_SortComponents
def get_config_from_kwds(**kwds):
    """Get an instance of IncidenceConfig from provided keyword arguments.

    If the ``method`` argument is ``IncidenceMethod.ampl_repn`` and no
    ``AMPLRepnVisitor`` has been provided, a new ``AMPLRepnVisitor`` is
    constructed. This function should generally be used by callers such
    as ``IncidenceGraphInterface`` to ensure that a visitor is created then
    re-used when calling ``get_incident_variables`` in a loop.

    """
    if kwds.get('method', None) is IncidenceMethod.ampl_repn and kwds.get('_ampl_repn_visitor', None) is None:
        subexpression_cache = {}
        subexpression_order = []
        external_functions = {}
        var_map = {}
        used_named_expressions = set()
        symbolic_solver_labels = False
        export_defined_variables = False
        sorter = FileDeterminism_to_SortComponents(FileDeterminism.ORDERED)
        amplvisitor = AMPLRepnVisitor(text_nl_template, subexpression_cache, subexpression_order, external_functions, var_map, used_named_expressions, symbolic_solver_labels, export_defined_variables, sorter)
        kwds['_ampl_repn_visitor'] = amplvisitor
    return IncidenceConfig(kwds)