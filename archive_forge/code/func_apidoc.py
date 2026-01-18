from __future__ import absolute_import, division, print_function
import errno
import glob
import json
import os
import re
import re
import re
@property
def apidoc(self):
    """
        The full apidoc.

        The apidoc will be fetched from the server, if that didn't happen yet.

        :returns: The apidoc.
        """
    if self._apidoc is None:
        self._apidoc = self._load_apidoc()
    return self._apidoc