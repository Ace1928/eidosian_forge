from __future__ import absolute_import, division, print_function
import os
import platform
import pwd
import re
import sys
import tempfile
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_bytes, to_native
from ansible.module_utils.six.moves import shlex_quote
def remove_job_file(self):
    try:
        os.unlink(self.cron_file)
        return True
    except OSError:
        return False
    except Exception:
        raise CronTabError('Unexpected error:', sys.exc_info()[0])