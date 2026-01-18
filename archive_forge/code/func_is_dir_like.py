from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import re
@property
def is_dir_like(self):
    if self.is_remote:
        return self.path.endswith('/')
    return self.path.endswith(os.sep)