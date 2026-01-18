import inspect
import re
import typing as T
from .common import (
def process_desc(desc: T.Optional[str], is_type: bool) -> str:
    if not desc:
        return ''
    if rendering_style == RenderingStyle.EXPANDED or (rendering_style == RenderingStyle.CLEAN and (not is_type)):
        first, *rest = desc.splitlines()
        return '\n'.join(['\n' + indent + first] + [indent + line for line in rest])
    first, *rest = desc.splitlines()
    return '\n'.join([' ' + first] + [indent + line for line in rest])