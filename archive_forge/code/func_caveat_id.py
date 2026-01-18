from base64 import standard_b64encode
from pymacaroons.utils import convert_to_string, convert_to_bytes
@caveat_id.setter
def caveat_id(self, value):
    self._caveat_id = convert_to_bytes(value)