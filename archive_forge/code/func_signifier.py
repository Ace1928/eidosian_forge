from pymacaroons.utils import convert_to_bytes
@signifier.setter
def signifier(self, string_or_bytes):
    self._signifier = convert_to_bytes(string_or_bytes)