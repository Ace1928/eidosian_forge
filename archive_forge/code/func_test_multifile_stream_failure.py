import warnings
from itertools import product
import numpy as np
import pytest
from ..filebasedimages import FileBasedHeader, FileBasedImage, SerializableImage
from .test_image_api import GenericImageAPI, SerializeMixin
def test_multifile_stream_failure():
    shape = (2, 3, 4)
    arr = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    img = SerializableMPNumpyImage(arr)
    with pytest.raises(NotImplementedError):
        img.to_bytes()
    img = SerializableNumpyImage(arr)
    bstr = img.to_bytes()
    with pytest.raises(NotImplementedError):
        SerializableMPNumpyImage.from_bytes(bstr)