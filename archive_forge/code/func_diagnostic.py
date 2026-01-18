import json
import os
import sys
from troveclient.compat import common
def diagnostic(self):
    """List diagnostic details about an instance."""
    self._require('id')
    self._pretty_print(self.dbaas.diagnostics.get, self.id)