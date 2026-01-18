from pprint import pformat
from six import iteritems
import re
@json_path.setter
def json_path(self, json_path):
    """
        Sets the json_path of this V1beta1CustomResourceColumnDefinition.
        JSONPath is a simple JSON path, i.e. with array notation.

        :param json_path: The json_path of this
        V1beta1CustomResourceColumnDefinition.
        :type: str
        """
    if json_path is None:
        raise ValueError('Invalid value for `json_path`, must not be `None`')
    self._json_path = json_path