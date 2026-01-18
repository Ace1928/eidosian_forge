from __future__ import (absolute_import, division, print_function)
import logging
import logging.config
import os
import tempfile
from datetime import datetime  # noqa: F401, pylint: disable=unused-import
from operator import eq
import time
from ansible.module_utils.common.text.converters import to_bytes
from ansible.module_utils.six import iteritems
def get_hashed_object(class_type, object_with_value, attributes_class_type=None, supported_attributes=None):
    """
    Convert any class instance into hashable so that the
    instances are eligible for various comparison
    operation available under set() object.
    :param class_type: Any class type whose instances needs to be hashable
    :param object_with_value: Instance of the class type with values which
     would be set in the resulting isinstance
    :param attributes_class_type: A list of class types of attributes, if attribute is a custom class instance
    :param supported_attributes: A list of attributes which should be considered while populating the instance
     with the values in the object. This helps in avoiding new attributes of the class_type which are still not
     supported by the current implementation.
    :return: A hashable instance with same state of the provided object_with_value
    """
    if object_with_value is None:
        return None
    HashedClass = generate_subclass(class_type)
    hashed_class_instance = HashedClass()
    if supported_attributes:
        class_attributes = list(set(hashed_class_instance.attribute_map) & set(supported_attributes))
    else:
        class_attributes = hashed_class_instance.attribute_map
    for attribute in class_attributes:
        attribute_value = getattr(object_with_value, attribute)
        if attributes_class_type:
            for attribute_class_type in attributes_class_type:
                if isinstance(attribute_value, attribute_class_type):
                    attribute_value = get_hashed_object(attribute_class_type, attribute_value)
        hashed_class_instance.__setattr__(attribute, attribute_value)
    return hashed_class_instance