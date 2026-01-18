from collections import ChainMap, Counter
from typing import Any, Dict, List, MutableMapping, Union
from typing import ChainMap as ChainMapType
from typing import Counter as CounterType
from ...errors import PdfReadError
from .. import mult
from ._font import Font
from ._text_state_params import TextStateParams
def reset_trm(self) -> TextStateManagerChainMapType:
    """Clear all transforms from chainmap having is_render==True"""
    while self.transform_stack.maps[0]['is_render']:
        self.transform_stack = self.transform_stack.parents
    return self.transform_stack