import abc
import binascii
from castellan.common import exception
def is_metadata_only(self):
    """Returns if the associated object is only metadata or not."""
    return self.get_encoded() is None