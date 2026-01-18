from typing import Dict, Any
from textwrap import indent
from ._list import render_list
from ..core import BaseRenderer, BlockState
from ..util import strip_end
def render_children(self, token, state: BlockState):
    children = token['children']
    return self.render_tokens(children, state)