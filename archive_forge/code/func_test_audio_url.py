import pathlib
from base64 import b64encode
from io import BytesIO
import numpy as np
import pytest
from panel.pane import Audio, Video
from panel.pane.media import TensorLike, _is_1dim_int_or_float_tensor
def test_audio_url(document, comm):
    audio_url = 'http://ccrma.stanford.edu/~jos/mp3/pno-cs.mp3'
    audio = Audio(audio_url)
    model = audio.get_root(document, comm=comm)
    assert audio_url == model.value