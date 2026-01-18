import ast
import sys
from typing import Any
from .internal import Filters, Key
def filters_to_expr(filter_obj: Any, is_root=True) -> str:
    op_map = {'>': '>', '<': '<', '=': '==', '==': '==', '!=': '!=', '>=': '>=', '<=': '<=', 'IN': 'in', 'NIN': 'not in', 'AND': 'and', 'OR': 'or'}

    def _convert_filter(filter: Any, is_root: bool) -> str:
        if hasattr(filter, 'filters') and filter.filters is not None:
            sub_expressions = [_convert_filter(f, False) for f in filter.filters if f.filters is not None or (f.key and f.key.name)]
            if not sub_expressions:
                return ''
            joint = ' and ' if filter.op == 'AND' else ' or '
            expr = joint.join(sub_expressions)
            return f'({expr})' if not is_root and sub_expressions else expr
        else:
            if not filter.key or not filter.key.name:
                return ''
            key_name = filter.key.name
            section = filter.key.section
            if section in section_map_reversed:
                function_name = section_map_reversed[section]
                key_name = f'{function_name}("{key_name}")'
            value = filter.value
            if value is None:
                value = 'None'
            elif isinstance(value, list):
                value = f'[{', '.join(map(str, value))}]'
            elif isinstance(value, str):
                value = f"'{value}'"
            return f'{key_name} {op_map[filter.op]} {value}'
    return _convert_filter(filter_obj, is_root)