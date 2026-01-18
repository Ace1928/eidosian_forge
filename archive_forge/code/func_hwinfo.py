import json
import os
import sys
from troveclient.compat import common
def hwinfo(self):
    """Show hardware information details about an instance."""
    self._require('id')
    self._pretty_print(self.dbaas.hwinfo.get, self.id)