import re
import unittest
from wsme import exc
from wsme import types
def test_file_field_storage(self):

    class buffer:

        def read(self):
            return 'from-file'

    class fieldstorage:
        filename = 'static.json'
        file = buffer()
        type = 'application/json'
    f = types.File(fieldstorage=fieldstorage)
    assert f.content == 'from-file'