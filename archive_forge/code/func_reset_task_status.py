import json
import os
import sys
from troveclient.compat import common
def reset_task_status(self):
    """Set the instance's task status to NONE."""
    self._require('id')
    self._pretty_print(self.dbaas.management.reset_task_status, self.id)