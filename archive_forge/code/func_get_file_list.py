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
def get_file_list(self):
    """Figure out the list of files to include in the source
        distribution, and put it in 'self.filelist'.  This might involve
        reading the manifest template (and writing the manifest), or just
        reading the manifest, or just using the default file set -- it all
        depends on the user's options.
        """
    template_exists = os.path.isfile(self.template)
    if not template_exists and self._manifest_is_not_generated():
        self.read_manifest()
        self.filelist.sort()
        self.filelist.remove_duplicates()
        return
    if not template_exists:
        self.warn(("manifest template '%s' does not exist " + '(using default file list)') % self.template)
    self.filelist.findall()
    if self.use_defaults:
        self.add_defaults()
    if template_exists:
        self.read_template()
    if self.prune:
        self.prune_file_list()
    self.filelist.sort()
    self.filelist.remove_duplicates()
    self.write_manifest()