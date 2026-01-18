from __future__ import absolute_import, division, print_function
import os
import os.path
import re
import shutil
import subprocess
import tempfile
import time
import shlex
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.parsing.convert_bool import BOOLEANS_FALSE
from ansible.module_utils.common.text.converters import to_text, to_bytes
def create_script(command):
    """Write out a script onto a target.

    This method should be backward compatible with Python when executing
    from within the container.

    :param command: command to run, this can be a script and can use spacing
                    with newlines as separation.
    :type command: ``str``
    """
    fd, script_file = tempfile.mkstemp(prefix='lxc-attach-script')
    f = os.fdopen(fd, 'wb')
    try:
        f.write(to_bytes(ATTACH_TEMPLATE % {'container_command': command}, errors='surrogate_or_strict'))
        f.flush()
    finally:
        f.close()
    os.chmod(script_file, int('0700', 8))
    stdout_file = os.fdopen(tempfile.mkstemp(prefix='lxc-attach-script-log')[0], 'ab')
    stderr_file = os.fdopen(tempfile.mkstemp(prefix='lxc-attach-script-err')[0], 'ab')
    try:
        subprocess.Popen([script_file], stdout=stdout_file, stderr=stderr_file).communicate()
    finally:
        stderr_file.close()
        stdout_file.close()
        os.remove(script_file)