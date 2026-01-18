from collections import ChainMap, Counter
from typing import Any, Dict, List, MutableMapping, Union
from typing import ChainMap as ChainMapType
from typing import Counter as CounterType
from ...errors import PdfReadError
from .. import mult
from ._font import Font
from ._text_state_params import TextStateParams
@property
def effective_transform(self) -> List[float]:
    """Current effective transform accounting for cm, tm, and trm transforms"""
    eff_transform = [*self.transform_stack.maps[0].values()]
    for transform in self.transform_stack.maps[1:]:
        eff_transform = mult(eff_transform, transform)
    return eff_transform