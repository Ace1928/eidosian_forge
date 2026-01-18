from collections import ChainMap, Counter
from typing import Any, Dict, List, MutableMapping, Union
from typing import ChainMap as ChainMapType
from typing import Counter as CounterType
from ...errors import PdfReadError
from .. import mult
from ._font import Font
from ._text_state_params import TextStateParams
@staticmethod
def raw_transform(_a: float=1.0, _b: float=0.0, _c: float=0.0, _d: float=1.0, _e: float=0.0, _f: float=0.0) -> Dict[int, float]:
    """Only a/b/c/d/e/f matrix params"""
    return dict(zip(range(6), map(float, (_a, _b, _c, _d, _e, _f))))