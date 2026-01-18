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
def prune_file_list(self):
    """Prune off branches that might slip into the file list as created
        by 'read_template()', but really don't belong there:
          * the build tree (typically "build")
          * the release tree itself (only an issue if we ran "sdist"
            previously with --keep-temp, or it aborted)
          * any RCS, CVS, .svn, .hg, .git, .bzr, _darcs directories
        """
    build = self.get_finalized_command('build')
    base_dir = self.distribution.get_fullname()
    self.filelist.exclude_pattern(None, prefix=build.build_base)
    self.filelist.exclude_pattern(None, prefix=base_dir)
    if sys.platform == 'win32':
        seps = '/|\\\\'
    else:
        seps = '/'
    vcs_dirs = ['RCS', 'CVS', '\\.svn', '\\.hg', '\\.git', '\\.bzr', '_darcs']
    vcs_ptrn = '(^|%s)(%s)(%s).*' % (seps, '|'.join(vcs_dirs), seps)
    self.filelist.exclude_pattern(vcs_ptrn, is_regex=1)