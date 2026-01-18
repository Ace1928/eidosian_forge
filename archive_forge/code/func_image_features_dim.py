from os.path import join
from typing import Dict, Optional, Tuple, Union
import onnxruntime as ort
from numpy import ndarray
@property
def image_features_dim(self) -> int:
    """Dimensionality of the image encoder features."""
    return self._image_encoder_dim