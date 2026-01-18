from typing import Optional
from .core import BlockState
from .block_parser import BlockParser
from .inline_parser import InlineParser
def render_state(self, state: BlockState):
    data = self._iter_render(state.tokens, state)
    if self.renderer:
        return self.renderer(data, state)
    return list(data)