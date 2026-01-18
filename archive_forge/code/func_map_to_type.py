from types import MappingProxyType
from email import utils
from email import errors
from email import _header_value_parser as parser
def map_to_type(self, name, cls):
    """Register cls as the specialized class for handling "name" headers.

        """
    self.registry[name.lower()] = cls