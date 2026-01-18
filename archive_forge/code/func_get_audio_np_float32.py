import pathlib
from base64 import b64encode
from io import BytesIO
import numpy as np
import pytest
from panel.pane import Audio, Video
from panel.pane.media import TensorLike, _is_1dim_int_or_float_tensor
def get_audio_np_float32(duration=0.01):
    return get_audio_np_float64(duration=duration).astype(np.float32)