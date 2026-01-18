from os.path import join
from typing import Dict, Optional, Tuple, Union
import onnxruntime as ort
from numpy import ndarray
@property
def embedding_dim(self) -> int:
    """Dimensionality of shared space embedding."""
    return self._embedding_dim