from __future__ import absolute_import, division, print_function
import errno
import glob
import json
import os
import re
import re
import re
@property
def params_in_path(self):
    """
        Params that can be passed in the path (URL) of the route.

        :returns: The params.
        """
    return [part[1:] for part in self.path.split('/') if part.startswith(':')]