from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import cmd
import shlex
from typing import List, Optional
from absl import flags
from pyglib import appcommands
import bq_utils
from frontend import bigquery_command
from frontend import bq_cached_client
def precmd(self, line: str) -> str:
    """Preprocess the shell input."""
    if line == 'EOF':
        return line
    if line.startswith('exit') or line.startswith('quit'):
        return 'EOF'
    words = line.strip().split()
    if len(words) > 1 and words[0].lower() == 'select':
        return 'query %s' % (shlex.quote(line),)
    if len(words) == 1 and words[0] not in ['help', 'ls', 'version']:
        return 'help %s' % (line.strip(),)
    return line