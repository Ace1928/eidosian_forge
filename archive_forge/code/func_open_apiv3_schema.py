from pprint import pformat
from six import iteritems
import re
@open_apiv3_schema.setter
def open_apiv3_schema(self, open_apiv3_schema):
    """
        Sets the open_apiv3_schema of this V1beta1CustomResourceValidation.
        OpenAPIV3Schema is the OpenAPI v3 schema to be validated against.

        :param open_apiv3_schema: The open_apiv3_schema of this
        V1beta1CustomResourceValidation.
        :type: V1beta1JSONSchemaProps
        """
    self._open_apiv3_schema = open_apiv3_schema