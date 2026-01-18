from base64 import standard_b64encode
from pymacaroons.utils import convert_to_string, convert_to_bytes
@property
def caveat_id_bytes(self):
    return self._caveat_id