from __future__ import absolute_import, division, print_function
import os
import re
import types
import copy
import inspect
import traceback
import json
from os.path import expanduser
from ansible.module_utils.basic import \
from ansible.module_utils.six.moves import configparser
import ansible.module_utils.six.moves.urllib.parse as urlparse
from base64 import b64encode, b64decode
from hashlib import sha256
from hmac import HMAC
from time import time
def has_tags(self, obj_tags, tag_list):
    """
        Used in fact modules to compare object tags to list of parameter tags. Return true if list of parameter tags
        exists in object tags.

        :param obj_tags: dictionary of tags from an Azure object.
        :param tag_list: list of tag keys or tag key:value pairs
        :return: bool
        """
    if not obj_tags and tag_list:
        return False
    if not tag_list:
        return True
    matches = 0
    result = False
    for tag in tag_list:
        tag_key = tag
        tag_value = None
        if ':' in tag:
            tag_key, tag_value = tag.split(':')
        if tag_value and obj_tags.get(tag_key) == tag_value:
            matches += 1
        elif not tag_value and obj_tags.get(tag_key):
            matches += 1
    if matches == len(tag_list):
        result = True
    return result