import json
import os
import sys
def dict_to_object(json_dict):
    """Converts a dictionary to a python object.

    Converts key-values to attribute-values.

    Args:
      json_dict: ({str: object, ...})

    Returns:
      (JSONObject)
    """
    obj = JSONObject()
    for name, val in json_dict.iteritems():
        if isinstance(val, dict):
            val = dict_to_object(val)
        setattr(obj, name, val)
    return obj