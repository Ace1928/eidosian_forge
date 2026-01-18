import bz2
import gzip
import os
import platform
import numpy as np
def download_command_wget(self):
    return 'wget --progress=bar:force -c -P %s %s' % (data_dir, self.url)