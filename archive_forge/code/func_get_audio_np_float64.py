import pathlib
from base64 import b64encode
from io import BytesIO
import numpy as np
import pytest
from panel.pane import Audio, Video
from panel.pane.media import TensorLike, _is_1dim_int_or_float_tensor
def get_audio_np_float64(duration=0.01):
    sample_rate = Audio.sample_rate
    time_variable = np.linspace(0, duration, int(duration * sample_rate), endpoint=False)
    sine_wave_400hz = 0.5 * np.sin(2 * np.pi * 440 * time_variable)
    return sine_wave_400hz