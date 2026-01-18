import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Union
from .. import mult, orient
from ._font import Font
def displaced_transform(self) -> List[float]:
    """Effective transform matrix after text has been rendered."""
    return mult(self.displacement_matrix(), self.transform)