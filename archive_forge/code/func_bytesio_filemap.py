from io import BytesIO
import numpy as np
from ..optpkg import optional_package
from numpy.testing import assert_array_equal
def bytesio_filemap(klass):
    """Return bytes io filemap for this image class `klass`"""
    file_map = klass.make_file_map()
    for name, fileholder in file_map.items():
        fileholder.fileobj = BytesIO()
        fileholder.pos = 0
    return file_map