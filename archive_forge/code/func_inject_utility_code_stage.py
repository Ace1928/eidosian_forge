from __future__ import absolute_import
import itertools
from time import time
from . import Errors
from . import DebugFlags
from . import Options
from .Errors import CompileError, InternalError, AbortError
from . import Naming
def inject_utility_code_stage(module_node):
    module_node.prepare_utility_code()
    use_utility_code_definitions(context.cython_scope, module_node.scope)
    utility_code_list = module_node.scope.utility_code_list
    utility_code_list[:] = sorted_utility_codes_and_deps(utility_code_list)
    normalize_deps(utility_code_list)
    added = set()
    for utilcode in utility_code_list:
        if utilcode in added:
            continue
        added.add(utilcode)
        if utilcode.requires:
            for dep in utilcode.requires:
                if dep not in added:
                    utility_code_list.append(dep)
        tree = utilcode.get_tree(cython_scope=context.cython_scope)
        if tree:
            module_node.merge_in(tree.with_compiler_directives(), tree.scope, stage='utility', merge_scope=True)
    return module_node