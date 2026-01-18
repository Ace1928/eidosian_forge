from __future__ import absolute_import, division, print_function
import errno
import glob
import json
import os
import re
import re
import re
@property
def apidoc_cache_file(self):
    """
        Full local path to the cached apidoc.
        """
    return os.path.join(self.apidoc_cache_dir, '{0}{1}'.format(self.apidoc_cache_name, self.cache_extension))