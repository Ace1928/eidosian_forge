from io import BytesIO
import numpy as np
from ..optpkg import optional_package
from numpy.testing import assert_array_equal
def bytesio_round_trip(img):
    """Save then load image from bytesio"""
    klass = img.__class__
    bytes_map = bytesio_filemap(klass)
    img.to_file_map(bytes_map)
    return klass.from_file_map(bytes_map)