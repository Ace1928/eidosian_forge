import os
import sys
from glob import glob
from warnings import warn
from distutils.core import Command
from distutils import dir_util
from distutils import file_util
from distutils import archive_util
from distutils.text_file import TextFile
from distutils.filelist import FileList
from distutils import log
from distutils.util import convert_path
from distutils.errors import DistutilsTemplateError, DistutilsOptionError
def read_manifest(self):
    """Read the manifest file (named by 'self.manifest') and use it to
        fill in 'self.filelist', the list of files to include in the source
        distribution.
        """
    log.info("reading manifest file '%s'", self.manifest)
    with open(self.manifest) as manifest:
        for line in manifest:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            self.filelist.append(line)