import logging
from typing import (
from typing_extensions import TypeAlias
def r_sa_check(template: str, tag_type: str, is_standalone: bool) -> bool:
    """Do a final checkto see if a tag could be a standalone"""
    if is_standalone and tag_type not in ['variable', 'no escape']:
        on_newline = template.split('\n', 1)
        if on_newline[0].isspace() or not on_newline[0]:
            return True
        else:
            return False
    else:
        return False