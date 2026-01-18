from collections import ChainMap, Counter
from typing import Any, Dict, List, MutableMapping, Union
from typing import ChainMap as ChainMapType
from typing import Counter as CounterType
from ...errors import PdfReadError
from .. import mult
from ._font import Font
from ._text_state_params import TextStateParams
def add_tm(self, operands: List[float]) -> TextStateManagerChainMapType:
    """Append a text transform matrix"""
    self.transform_stack = self.transform_stack.new_child(self.new_transform(*self._complete_matrix(operands), is_text=True))
    return self.transform_stack