from dataclasses import dataclass
from typing import Dict, Optional, Tuple
def merge_operator_dicts(lhs: Dict[str, SelectiveBuildOperator], rhs: Dict[str, SelectiveBuildOperator]) -> Dict[str, SelectiveBuildOperator]:
    operators: Dict[str, SelectiveBuildOperator] = {}
    for op_name, op in list(lhs.items()) + list(rhs.items()):
        new_op = op
        if op_name in operators:
            new_op = combine_operators(operators[op_name], op)
        operators[op_name] = new_op
    return operators