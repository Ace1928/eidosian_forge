import logging
import os
import re
import string
import typing
from itertools import chain as _chain
def python_entrypoint_reference(value: str) -> bool:
    module, _, rest = value.partition(':')
    if '[' in rest:
        obj, _, extras_ = rest.partition('[')
        if extras_.strip()[-1] != ']':
            return False
        extras = (x.strip() for x in extras_.strip(string.whitespace + '[]').split(','))
        if not all((pep508_identifier(e) for e in extras)):
            return False
        _logger.warning(f'`{value}` - using extras for entry points is not recommended')
    else:
        obj = rest
    module_parts = module.split('.')
    identifiers = _chain(module_parts, obj.split('.')) if rest else module_parts
    return all((python_identifier(i.strip()) for i in identifiers))