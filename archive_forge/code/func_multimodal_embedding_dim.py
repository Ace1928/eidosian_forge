from os.path import join
from typing import Dict, Optional, Tuple, Union
import onnxruntime as ort
from numpy import ndarray
@property
def multimodal_embedding_dim(self) -> int:
    """Dimensionality of multimodal joint embedding."""
    return self._text_encoder_dim