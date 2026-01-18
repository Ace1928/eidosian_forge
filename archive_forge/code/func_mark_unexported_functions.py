from pythran.analyses import ExtendedSyntaxCheck
from pythran.optimizations import (ComprehensionPatterns, ListCompToGenexp,
from pythran.transformations import (ExpandBuiltins, ExpandImports,
import re
def mark_unexported_functions(ir, exported_functions):
    from pythran.metadata import add as MDadd, Local as MDLocal
    for node in ir.body:
        if hasattr(node, 'name'):
            if node.name not in exported_functions:
                MDadd(node, MDLocal())
    return ir