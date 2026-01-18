from __future__ import absolute_import, division, print_function
from ansible_collections.google.cloud.plugins.module_utils.gcp_utils import (
import json
import re
import time
def metadata_encoder(metadata):
    metadata_new = []
    for key in metadata:
        value = metadata[key]
        metadata_new.append({'key': key, 'value': value})
    return {'items': metadata_new}