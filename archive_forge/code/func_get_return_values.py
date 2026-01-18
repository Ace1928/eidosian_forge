from parso.python import tree
from jedi import debug
from jedi.inference.cache import inference_state_method_cache, CachedMetaClass
from jedi.inference import compiled
from jedi.inference import recursion
from jedi.inference import docstrings
from jedi.inference import flow_analysis
from jedi.inference.signature import TreeSignature
from jedi.inference.filters import ParserTreeFilter, FunctionExecutionFilter, \
from jedi.inference.names import ValueName, AbstractNameDefinition, \
from jedi.inference.base_value import ContextualizedNode, NO_VALUES, \
from jedi.inference.lazy_value import LazyKnownValues, LazyKnownValue, \
from jedi.inference.context import ValueContext, TreeContextMixin
from jedi.inference.value import iterable
from jedi import parser_utils
from jedi.inference.parser_cache import get_yield_exprs
from jedi.inference.helpers import values_from_qualified_names
from jedi.inference.gradual.generics import TupleGenericManager
@inference_state_method_cache(default=NO_VALUES)
@recursion.execution_recursion_decorator()
def get_return_values(self, check_yields=False):
    funcdef = self.tree_node
    if funcdef.type == 'lambdef':
        return self.infer_node(funcdef.children[-1])
    if check_yields:
        value_set = NO_VALUES
        returns = get_yield_exprs(self.inference_state, funcdef)
    else:
        value_set = self.infer_annotations()
        if value_set:
            return value_set
        value_set |= docstrings.infer_return_types(self._value)
        returns = funcdef.iter_return_stmts()
    for r in returns:
        if check_yields:
            value_set |= ValueSet.from_sets((lazy_value.infer() for lazy_value in self._get_yield_lazy_value(r)))
        else:
            check = flow_analysis.reachability_check(self, funcdef, r)
            if check is flow_analysis.UNREACHABLE:
                debug.dbg('Return unreachable: %s', r)
            else:
                try:
                    children = r.children
                except AttributeError:
                    ctx = compiled.builtin_from_name(self.inference_state, 'None')
                    value_set |= ValueSet([ctx])
                else:
                    value_set |= self.infer_node(children[1])
            if check is flow_analysis.REACHABLE:
                debug.dbg('Return reachable: %s', r)
                break
    return value_set