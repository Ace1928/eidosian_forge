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
def emptyline(self):
    print('Available commands:', end=' ')
    print(' '.join(list(self._commands)))