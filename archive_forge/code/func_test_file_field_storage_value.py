import re
import unittest
from wsme import exc
from wsme import types
def test_file_field_storage_value(self):

    class buffer:

        def read(self):
            return 'from-file'

    class fieldstorage:
        filename = 'static.json'
        file = None
        type = 'application/json'
        value = 'from-value'
    f = types.File(fieldstorage=fieldstorage)
    assert f.content == 'from-value'