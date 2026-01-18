from __future__ import absolute_import
import six
import copy
from collections import OrderedDict
from googleapiclient import _helpers as util
def prettyPrintSchema(self, schema):
    """Get pretty printed object prototype of schema.

    Args:
      schema: object, Parsed JSON schema.

    Returns:
      string, A string that contains a prototype object with
        comments that conforms to the given schema.
    """
    return self._prettyPrintSchema(schema, dent=0)[:-2]