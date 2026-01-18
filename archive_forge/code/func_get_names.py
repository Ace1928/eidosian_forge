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
def get_names(self) -> List[str]:
    names = dir(self)
    commands = (name for name in self._commands if name not in self._special_command_names)
    names.extend(('do_%s' % (name,) for name in commands))
    names.append('do_select')
    names.remove('do_EOF')
    return names