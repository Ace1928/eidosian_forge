from __future__ import (absolute_import, division, print_function)
from copy import deepcopy
import json
import re
import base64
import time
def is_label_value_changed(self, current_tags, desired_tags):
    tag_keys = list(current_tags.keys())
    user_tag_keys = [key for key in tag_keys if key not in ('count-down', 'gcp_resource_id', 'partner-platform-serial-number')]
    desired_keys = [a_dict['label_key'] for a_dict in desired_tags]
    if user_tag_keys == desired_keys:
        for tag in desired_tags:
            if current_tags[tag['label_key']] != tag['label_value']:
                return True
        return False
    else:
        return True