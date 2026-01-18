import re
import sys
import textwrap
from io import StringIO
from .. import commands, export_pot, option, registry, tests
def test_classes_nested(self):
    src = '\nclass Matroska(object):\n    class Smaller(object):\n        class Smallest(object):\n            pass\n'
    cls_lines, _ = export_pot._parse_source(src)
    self.assertEqual(cls_lines, {'Matroska': 2, 'Smaller': 3, 'Smallest': 4})