import gast as ast
from collections import defaultdict
from functools import reduce
from pythran.analyses import Aliases, CFG
from pythran.intrinsic import Intrinsic
from pythran.passmanager import ModuleAnalysis
from pythran.interval import Interval, IntervalTuple, UNKNOWN_RANGE
from pythran.tables import MODULES, attributes
def bound_range(mapping, aliases, node, modified=None):
    """
    Bound the identifier in `mapping' with the expression in `node'.
    `aliases' is the result of aliasing analysis and `modified' is
    updated with the set of identifiers possibly `bounded' as the result
    of the call.

    Returns `modified' or a fresh set of modified identifiers.

    """
    if modified is None:
        modified = set()
    if isinstance(node, ast.Name):
        pass
    elif isinstance(node, ast.UnaryOp):
        try:
            negated = negate(node.operand)
            bound_range(mapping, aliases, negated, modified)
        except UnsupportedExpression:
            pass
    elif isinstance(node, ast.BoolOp):
        if isinstance(node.op, ast.And):
            for value in node.values:
                bound_range(mapping, aliases, value, modified)
        elif isinstance(node.op, ast.Or):
            mappings = [mapping.copy() for _ in node.values]
            for value, mapping_cpy in zip(node.values, mappings):
                bound_range(mapping_cpy, aliases, value, modified)
            for k in modified:
                mapping[k] = reduce(lambda x, y: x.union(y[k]), mappings[1:], mappings[0][k])
    elif isinstance(node, ast.Compare):
        left = node.left
        if isinstance(node.left, ast.Name):
            modified.add(node.left.id)
        for op, right in zip(node.ops, node.comparators):
            if isinstance(right, ast.Name):
                modified.add(right.id)
            if isinstance(left, ast.Name):
                left_interval = mapping[left.id]
            else:
                left_interval = mapping[left]
            if isinstance(right, ast.Name):
                right_interval = mapping[right.id]
            else:
                right_interval = mapping[right]
            l_l, l_h = (left_interval.low, left_interval.high)
            r_l, r_h = (right_interval.low, right_interval.high)
            r_i = l_i = None
            if isinstance(op, ast.Eq):
                low, high = (max(l_l, r_l), min(l_h, r_h))
                if low <= high:
                    l_i = r_i = Interval(max(l_l, r_l), min(l_h, r_h))
            elif isinstance(op, ast.Lt):
                l_i = Interval(min(l_l, r_h - 1), min(l_h, r_h - 1))
                r_i = Interval(max(r_l, l_l + 1), max(r_h, l_l + 1))
            elif isinstance(op, ast.LtE):
                l_i = Interval(min(l_l, r_h), min(l_h, r_h))
                r_i = Interval(max(r_l, l_l), max(r_h, l_l))
            elif isinstance(op, ast.Gt):
                l_i = Interval(max(l_l, r_l + 1), max(l_h, r_l + 1))
                r_i = Interval(min(r_l, l_h - 1), min(r_h, l_h - 1))
            elif isinstance(op, ast.GtE):
                l_i = Interval(max(l_l, r_l), max(l_h, r_l))
                r_i = Interval(min(r_l, l_h), min(r_h, l_h))
            elif isinstance(op, ast.In):
                if isinstance(right, (ast.List, ast.Tuple, ast.Set)):
                    if right.elts:
                        low = min((mapping[elt].low for elt in right.elts))
                        high = max((mapping[elt].high for elt in right.elts))
                        l_i = Interval(low, high)
                elif isinstance(right, ast.Call):
                    for alias in aliases[right.func]:
                        if not hasattr(alias, 'return_range_content'):
                            l_i = None
                            break
                        rrc = alias.return_range_content([mapping[arg] for arg in right.args])
                        if l_i is None:
                            l_i = rrc
                        else:
                            l_i = l_i.union(alias.return_range(right))
            if l_i is not None and isinstance(left, ast.Name):
                mapping[left.id] = l_i
            if r_i is not None and isinstance(right, ast.Name):
                mapping[right.id] = r_i
            left = right