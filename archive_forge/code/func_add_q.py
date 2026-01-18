from collections import ChainMap, Counter
from typing import Any, Dict, List, MutableMapping, Union
from typing import ChainMap as ChainMapType
from typing import Counter as CounterType
from ...errors import PdfReadError
from .. import mult
from ._font import Font
from ._text_state_params import TextStateParams
def add_q(self) -> None:
    """Add another level to q_queue"""
    self.q_depth.append(len(self.q_depth))