from pprint import pformat
from six import iteritems
import re
@example.setter
def example(self, example):
    """
        Sets the example of this V1beta1JSONSchemaProps.
        JSON represents any valid JSON value. These types are supported: bool,
        int64, float64, string, []interface{}, map[string]interface{} and nil.

        :param example: The example of this V1beta1JSONSchemaProps.
        :type: object
        """
    self._example = example