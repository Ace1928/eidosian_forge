from io import BytesIO
import breezy.bzr.xml5
from ... import errors, fifo_cache
from .. import inventory, serializer, xml6, xml7, xml8
from ..inventory import Inventory
from . import TestCase
def test_empty_property_value(self):
    """Create an empty property value check that it serializes correctly"""
    s_v5 = breezy.bzr.xml5.serializer_v5
    rev = s_v5.read_revision_from_string(_revision_v5)
    props = {'empty': '', 'one': 'one'}
    rev.properties = props
    txt = b''.join(s_v5.write_revision_to_lines(rev))
    new_rev = s_v5.read_revision_from_string(txt)
    self.assertEqual(props, new_rev.properties)