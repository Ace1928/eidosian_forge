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
def make_release_tree(self, base_dir, files):
    """Create the directory tree that will become the source
        distribution archive.  All directories implied by the filenames in
        'files' are created under 'base_dir', and then we hard link or copy
        (if hard linking is unavailable) those files into place.
        Essentially, this duplicates the developer's source tree, but in a
        directory named after the distribution, containing only the files
        to be distributed.
        """
    self.mkpath(base_dir)
    dir_util.create_tree(base_dir, files, dry_run=self.dry_run)
    if hasattr(os, 'link'):
        link = 'hard'
        msg = 'making hard links in %s...' % base_dir
    else:
        link = None
        msg = 'copying files to %s...' % base_dir
    if not files:
        log.warn('no files to distribute -- empty manifest?')
    else:
        log.info(msg)
    for file in files:
        if not os.path.isfile(file):
            log.warn("'%s' not a regular file -- skipping", file)
        else:
            dest = os.path.join(base_dir, file)
            self.copy_file(file, dest, link=link)
    self.distribution.metadata.write_pkg_info(base_dir)