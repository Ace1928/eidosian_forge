import unittest
from array import array
import binascii
from .crcmod import mkCrcFun, Crc
from .crcmod import _usingExtension
from .predefined import PredefinedCrc
from .predefined import mkPredefinedCrcFun
from .predefined import _crc_definitions as _predefined_crc_definitions
def test_function_predefined_table(self):
    for table_entry in _predefined_crc_definitions:
        crc_func = mkPredefinedCrcFun(table_entry['name'])
        calc_value = crc_func(b'123456789')
        self.assertEqual(calc_value, table_entry['check'], "Wrong answer for CRC '%s'" % table_entry['name'])