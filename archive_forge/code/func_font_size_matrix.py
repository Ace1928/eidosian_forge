import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Union
from .. import mult, orient
from ._font import Font
def font_size_matrix(self) -> List[float]:
    """Font size matrix"""
    return [self.font_size * (self.Tz / 100.0), 0.0, 0.0, self.font_size, 0.0, self.Ts]