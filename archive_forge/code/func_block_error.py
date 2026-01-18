from typing import Dict, Any
from textwrap import indent
from ._list import render_list
from ..core import BaseRenderer, BlockState
from ..util import strip_end
def block_error(self, token: Dict[str, Any], state: BlockState) -> str:
    return ''