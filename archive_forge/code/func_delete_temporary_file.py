from __future__ import absolute_import, division, print_function
import os
import os.path
import shutil
import subprocess
import tempfile
import json
from ansible.parsing.yaml.loader import AnsibleLoader
from ansible.errors import AnsibleError, AnsibleFileNotFound
from ansible.module_utils.six.moves import shlex_quote
from ansible.module_utils._text import to_bytes
from ansible.plugins.connection import ConnectionBase, BUFSIZE
from ansible.utils.display import Display
def delete_temporary_file(self):
    if self._file_to_delete is not None:
        os.remove(self._file_to_delete)
        self._file_to_delete = None