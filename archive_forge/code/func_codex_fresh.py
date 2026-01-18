from __future__ import absolute_import, division, print_function
import datetime
import fileinput
import os
import re
import shutil
import sys
from ansible.module_utils.basic import AnsibleModule
def codex_fresh(codex, module):
    """ Check if grimoire collection is fresh enough. """
    if not module.params['cache_valid_time']:
        return False
    timedelta = datetime.timedelta(seconds=module.params['cache_valid_time'])
    for grimoire in codex:
        lastupdate_path = os.path.join(SORCERY_STATE_DIR, grimoire + '.lastupdate')
        try:
            mtime = os.stat(lastupdate_path).st_mtime
        except Exception:
            return False
        lastupdate_ts = datetime.datetime.fromtimestamp(mtime)
        if lastupdate_ts + timedelta < datetime.datetime.now():
            return False
    return True