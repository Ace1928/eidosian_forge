import io
import os
import pathlib
import re
import sys
from pprint import pformat
from IPython.core import magic_arguments
from IPython.core import oinspect
from IPython.core import page
from IPython.core.alias import AliasError, Alias
from IPython.core.error import UsageError
from IPython.core.magic import  (
from IPython.testing.skipdoctest import skip_doctest
from IPython.utils.openpy import source_to_unicode
from IPython.utils.process import abbrev_cwd
from IPython.utils.terminal import set_term_title
from traitlets import Bool
from warnings import warn
@line_magic
def rehashx(self, parameter_s=''):
    """Update the alias table with all executable files in $PATH.

        rehashx explicitly checks that every entry in $PATH is a file
        with execute access (os.X_OK).

        Under Windows, it checks executability as a match against a
        '|'-separated string of extensions, stored in the IPython config
        variable win_exec_ext.  This defaults to 'exe|com|bat'.

        This function also resets the root module cache of module completer,
        used on slow filesystems.
        """
    from IPython.core.alias import InvalidAliasError
    del self.shell.db['rootmodules_cache']
    path = [os.path.abspath(os.path.expanduser(p)) for p in os.environ.get('PATH', '').split(os.pathsep)]
    syscmdlist = []
    savedir = os.getcwd()
    try:
        if self.is_posix:
            for pdir in path:
                try:
                    os.chdir(pdir)
                except OSError:
                    continue
                dirlist = os.scandir(path=pdir)
                for ff in dirlist:
                    if self.isexec(ff):
                        fname = ff.name
                        try:
                            if not self.shell.alias_manager.is_alias(fname):
                                self.shell.alias_manager.define_alias(fname.replace('.', ''), fname)
                        except InvalidAliasError:
                            pass
                        else:
                            syscmdlist.append(fname)
        else:
            no_alias = Alias.blacklist
            for pdir in path:
                try:
                    os.chdir(pdir)
                except OSError:
                    continue
                dirlist = os.scandir(pdir)
                for ff in dirlist:
                    fname = ff.name
                    base, ext = os.path.splitext(fname)
                    if self.isexec(ff) and base.lower() not in no_alias:
                        if ext.lower() == '.exe':
                            fname = base
                            try:
                                self.shell.alias_manager.define_alias(base.lower().replace('.', ''), fname)
                            except InvalidAliasError:
                                pass
                            syscmdlist.append(fname)
        self.shell.db['syscmdlist'] = syscmdlist
    finally:
        os.chdir(savedir)