from base64 import standard_b64encode
from pymacaroons.utils import convert_to_string, convert_to_bytes
def first_party(self):
    return self._verification_key_id is None