from pythran.analyses import LazynessAnalysis, StrictAliases, YieldPoints
from pythran.analyses import LocalNodeDeclarations, Immediates, RangeValues
from pythran.analyses import Ancestors
from pythran.config import cfg
from pythran.cxxtypes import TypeBuilder, ordered_set
from pythran.intrinsic import UserFunction, Class
from pythran.passmanager import ModuleAnalysis
from pythran.tables import operator_to_lambda, MODULES
from pythran.types.conversion import pytype_to_ctype
from pythran.types.reorder import Reorder
from pythran.utils import attr_to_path, cxxid, isnum, isextslice
from collections import defaultdict
from functools import reduce
import gast as ast
from itertools import islice
from copy import deepcopy
def combine_(self, node, op, othernode):
    try:
        try:
            node_id, depth = self.node_to_id(node)
            if depth:
                node = ast.Name(node_id, ast.Load(), None, None)
                former_op = op

                def merge_container_type(ty, index):
                    if isinstance(index, int):
                        kty = self.builder.NamedType('std::integral_constant<long,{}>'.format(index))
                        return self.builder.IndexableContainerType(kty, ty)
                    elif isinstance(index, float):
                        kty = self.builder.NamedType('double')
                        return self.builder.IndexableContainerType(kty, ty)
                    else:
                        return self.builder.ContainerType(ty)

                def op(*args):
                    return reduce(merge_container_type, depth, former_op(*args))
            self.name_to_nodes[node_id].append(node)
        except UnboundableRValue:
            pass
        if self.isargument(node):
            node_id, _ = self.node_to_id(node)
            if node not in self.result:
                self.result[node] = op(self.result[othernode])
            assert self.result[node], 'found an alias with a type'
            parametric_type = self.builder.PType(self.current, self.result[othernode])
            if self.register(self.current, node_id, parametric_type):
                current_function = self.combiners[self.current]

                def translator_generator(args, op):
                    """ capture args for translator generation"""

                    def interprocedural_type_translator(s, n):
                        translated_othernode = ast.Name('__fake__', ast.Load(), None, None)
                        s.result[translated_othernode] = parametric_type.instanciate(s.current, [s.result[arg] for arg in n.args])
                        for p, effective_arg in enumerate(n.args):
                            formal_arg = args[p]
                            if formal_arg.id == node_id:
                                translated_node = effective_arg
                                break
                        try:
                            s.combine(translated_node, op, translated_othernode)
                        except NotImplementedError:
                            pass
                        except UnboundLocalError:
                            pass
                    return interprocedural_type_translator
                translator = translator_generator(self.current.args.args, op)
                current_function.add_combiner(translator)
        else:
            self.update_type(node, op, self.result[othernode])
    except UnboundableRValue:
        pass