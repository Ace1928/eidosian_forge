from __future__ import absolute_import, division, print_function
import errno
import glob
import json
import os
import re
import re
import re
@property
def cache_extension(self):
    """
        File extension for the local cache file.

        Will include the language if set.
        """
    if self.language:
        ext = '.{}.json'.format(self.language)
    else:
        ext = '.json'
    return ext