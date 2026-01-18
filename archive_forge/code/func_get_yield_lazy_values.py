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
@recursion.execution_recursion_decorator(default=iter([]))
def get_yield_lazy_values(self, is_async=False):
    for_parents = [(y, tree.search_ancestor(y, 'for_stmt', 'funcdef', 'while_stmt', 'if_stmt')) for y in get_yield_exprs(self.inference_state, self.tree_node)]
    yields_order = []
    last_for_stmt = None
    for yield_, for_stmt in for_parents:
        parent = for_stmt.parent
        if parent.type == 'suite':
            parent = parent.parent
        if for_stmt.type == 'for_stmt' and parent == self.tree_node and parser_utils.for_stmt_defines_one_name(for_stmt):
            if for_stmt == last_for_stmt:
                yields_order[-1][1].append(yield_)
            else:
                yields_order.append((for_stmt, [yield_]))
        elif for_stmt == self.tree_node:
            yields_order.append((None, [yield_]))
        else:
            types = self.get_return_values(check_yields=True)
            if types:
                yield LazyKnownValues(types, min=0, max=float('inf'))
            return
        last_for_stmt = for_stmt
    for for_stmt, yields in yields_order:
        if for_stmt is None:
            for yield_ in yields:
                yield from self._get_yield_lazy_value(yield_)
        else:
            input_node = for_stmt.get_testlist()
            cn = ContextualizedNode(self, input_node)
            ordered = cn.infer().iterate(cn)
            ordered = list(ordered)
            for lazy_value in ordered:
                dct = {str(for_stmt.children[1].value): lazy_value.infer()}
                with self.predefine_names(for_stmt, dct):
                    for yield_in_same_for_stmt in yields:
                        yield from self._get_yield_lazy_value(yield_in_same_for_stmt)