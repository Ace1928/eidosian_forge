from pprint import pformat
from six import iteritems
import re
@binary_data.setter
def binary_data(self, binary_data):
    """
        Sets the binary_data of this V1ConfigMap.
        BinaryData contains the binary data. Each key must consist of
        alphanumeric characters, '-', '_' or '.'. BinaryData can contain byte
        sequences that are not in the UTF-8 range. The keys stored in BinaryData
        must not overlap with the ones in the Data field, this is enforced
        during validation process. Using this field will require 1.10+ apiserver
        and kubelet.

        :param binary_data: The binary_data of this V1ConfigMap.
        :type: dict(str, str)
        """
    self._binary_data = binary_data