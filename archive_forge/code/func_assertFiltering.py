from io import BytesIO
from unittest import TestCase
from fastimport import (
from fastimport.processors import (
from :2
from :2
from :100
from :101
from :100
from :100
from :100
from :100
from :101
from :100
from :100
from :102
from :102
from :102
from :100
from :102
from :100
from :102
from :100
from :102
from :102
from :102
from :100
from :102
from :100
from :100
from :100
from :100
from :100
from :102
from :101
from :102
from :101
import
from :999
from :3
import
from :999
from :3
import
from :999
from :3
import
from :999
from :3
import
from :999
from :3
def assertFiltering(self, input_stream, params, expected):
    outf = BytesIO()
    proc = filter_processor.FilterProcessor(params=params)
    proc.outf = outf
    s = BytesIO(input_stream)
    p = parser.ImportParser(s)
    proc.process(p.iter_commands)
    out = outf.getvalue()
    self.assertEqual(expected, out)