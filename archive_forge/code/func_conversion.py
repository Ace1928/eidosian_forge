from pprint import pformat
from six import iteritems
import re
@conversion.setter
def conversion(self, conversion):
    """
        Sets the conversion of this V1beta1CustomResourceDefinitionSpec.
        `conversion` defines conversion settings for the CRD.

        :param conversion: The conversion of this
        V1beta1CustomResourceDefinitionSpec.
        :type: V1beta1CustomResourceConversion
        """
    self._conversion = conversion