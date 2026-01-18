from __future__ import (absolute_import, division, print_function)
import json
import re
from datetime import timedelta
from ansible.module_utils.basic import env_fallback
from ansible.module_utils.common.collections import is_sequence
from ansible.module_utils.six.moves.urllib.parse import urlparse
def is_image_name_id(name):
    """Check whether the given image name is in fact an image ID (hash)."""
    if re.match('^sha256:[0-9a-fA-F]{64}$', name):
        return True
    return False