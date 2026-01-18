from __future__ import annotations
import argparse
from typing import Any, Dict, List, Optional, Tuple, Union, Type, Set, TYPE_CHECKING
def parse_one(name: str, field: FieldInfo) -> Tuple[Set, Dict]:
    """
    Parse a single field into the args and kwargs for the add_argument method
    """
    args = set()
    kwargs = {}
    field_type = field.annotation
    for pt in _parser_types.values():
        if field_type == Optional[pt]:
            field_type = pt
            break
    if field.is_required():
        args.add(name)
    else:
        args.add(f'--{name}')
        if field.json_schema_extra and field.json_schema_extra.get('alt'):
            alt = field.json_schema_extra['alt']
            if isinstance(alt, str):
                alt = [alt]
            for a in alt:
                if not isinstance(a, str):
                    a = str(a)
                if not a.startswith('-'):
                    a = f'-{a}'
                args.add(a)
        elif len(name) > 3:
            if '_' in name:
                parts = name.split('_')
                short_name = ''.join([p[0] for p in parts])
                args.add(f'-{short_name}')
            else:
                args.add(f'-{name[:2]}')
    if field_type == bool:
        kwargs['action'] = 'store_true' if field.default is False or field.default is None else 'store_false'
    elif field_type in {list[str], list[int], list[float], List[str], List[int], List[float]}:
        kwargs['action'] = 'append'
    else:
        kwargs['type'] = field_type
    if field.default is not None:
        kwargs['default'] = field.default
    elif field.default_factory is not None:
        kwargs['default'] = None
    if field.description is not None:
        kwargs['help'] = field.description
    return (args, kwargs)