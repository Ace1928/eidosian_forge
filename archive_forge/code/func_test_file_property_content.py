import re
import unittest
from wsme import exc
from wsme import types
def test_file_property_content(self):

    class buffer:

        def read(self):
            return 'from-file'
    f = types.File(content=b'from-content')
    assert f.file.read() == b'from-content'