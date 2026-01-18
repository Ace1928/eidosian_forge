from __future__ import (absolute_import, division, print_function)
from copy import deepcopy
import json
import re
import base64
import time
def compare_gcp_labels(self, current_tags, user_tags, is_ha):
    """
        Update user-tag API behaves differently in GCP CVO.
        It only supports adding gcp_labels and modifying the values of gcp_labels. Removing gcp_label is not allowed.
        """
    resp, error = self.user_tag_key_unique(user_tags, 'label_key')
    if error is not None:
        return (None, error)
    resp, error = self.current_label_exist(current_tags, user_tags, is_ha)
    if error is not None:
        return (None, error)
    if self.is_label_value_changed(current_tags, user_tags):
        return (True, None)
    else:
        return (None, None)