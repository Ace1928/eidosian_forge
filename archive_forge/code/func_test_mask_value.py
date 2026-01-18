import uuid
from pycadf.tests import base
from pycadf import utils
def test_mask_value(self):
    value = str(uuid.uuid4())
    m_percent = 0.125
    obfuscate = utils.mask_value(value, m_percent)
    visible = int(round(len(value) * m_percent))
    self.assertEqual(value[:visible], obfuscate[:visible])
    self.assertNotEqual(value[:visible + 1], obfuscate[:visible + 1])
    self.assertEqual(value[-visible:], obfuscate[-visible:])
    self.assertNotEqual(value[-visible - 1:], obfuscate[-visible - 1:])