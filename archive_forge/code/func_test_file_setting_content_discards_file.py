import re
import unittest
from wsme import exc
from wsme import types
def test_file_setting_content_discards_file(self):

    class buffer:

        def read(self):
            return 'from-file'
    f = types.File(file=buffer())
    f.content = 'from-content'
    assert f.content == 'from-content'