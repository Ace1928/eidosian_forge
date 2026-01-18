import logging
from typing import (
from typing_extensions import TypeAlias
def grab_literal(template: str, l_del: str) -> Tuple[str, str]:
    """Parse a literal from the template"""
    global _CURRENT_LINE
    try:
        literal, template = template.split(l_del, 1)
        _CURRENT_LINE += literal.count('\n')
        return (literal, template)
    except ValueError:
        return (template, '')