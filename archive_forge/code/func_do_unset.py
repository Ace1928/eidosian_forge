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
def do_unset(self, line: str) -> int:
    """Unset the value of the project_id or dataset_id flag."""
    name = line.strip()
    client = bq_cached_client.Client.Get()
    if name not in ('project_id', 'dataset_id'):
        print('unset (project_id|dataset_id)')
    else:
        setattr(client, name, '')
        if name == 'project_id':
            client.dataset_id = ''
        self._set_prompt()
    return 0