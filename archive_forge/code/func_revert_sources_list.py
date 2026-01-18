from __future__ import absolute_import, division, print_function
import copy
import glob
import json
import os
import re
import sys
import tempfile
import random
import time
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.respawn import has_respawned, probe_interpreters_for_module, respawn_module
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.six import PY3
from ansible.module_utils.urls import fetch_url
from ansible.module_utils.common.locale import get_best_parsable_locale
def revert_sources_list(sources_before, sources_after, sourceslist_before):
    """Revert the sourcelist files to their previous state."""
    for filename in set(sources_after.keys()).difference(sources_before.keys()):
        if os.path.exists(filename):
            os.remove(filename)
    sourceslist_before.save()