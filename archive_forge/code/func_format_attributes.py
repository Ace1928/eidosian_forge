from __future__ import (absolute_import, division, print_function)
import os
import stat
import re
def format_attributes(attributes):
    attribute_list = [FILE_ATTRIBUTES.get(attr) for attr in attributes if attr in FILE_ATTRIBUTES]
    return attribute_list