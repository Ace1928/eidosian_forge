from pprint import pformat
from six import iteritems
import re
@additional_properties.setter
def additional_properties(self, additional_properties):
    """
        Sets the additional_properties of this V1beta1JSONSchemaProps.
        JSONSchemaPropsOrBool represents JSONSchemaProps or a boolean value.
        Defaults to true for the boolean property.

        :param additional_properties: The additional_properties of this
        V1beta1JSONSchemaProps.
        :type: object
        """
    self._additional_properties = additional_properties