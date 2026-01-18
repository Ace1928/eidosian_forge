from pprint import pformat
from six import iteritems
import re
@field_path.setter
def field_path(self, field_path):
    """
        Sets the field_path of this V1ObjectFieldSelector.
        Path of the field to select in the specified API version.

        :param field_path: The field_path of this V1ObjectFieldSelector.
        :type: str
        """
    if field_path is None:
        raise ValueError('Invalid value for `field_path`, must not be `None`')
    self._field_path = field_path