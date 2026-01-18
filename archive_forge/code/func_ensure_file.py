import pkg_resources
import sys
import optparse
from . import bool_optparse
import os
import re
import textwrap
from . import pluginlib
import configparser
import getpass
from logging.config import fileConfig
def ensure_file(self, filename, content, svn_add=True):
    """
        Ensure a file named ``filename`` exists with the given
        content.  If ``--interactive`` has been enabled, this will ask
        the user what to do if a file exists with different content.
        """
    global difflib
    assert content is not None, 'You cannot pass a content of None'
    self.ensure_dir(os.path.dirname(filename), svn_add=svn_add)
    if not os.path.exists(filename):
        if self.verbose:
            print('Creating %s' % filename)
        if not self.simulate:
            f = open(filename, 'wb')
            f.write(content)
            f.close()
        if svn_add and os.path.exists(os.path.join(os.path.dirname(filename), '.svn')):
            self.svn_command('add', filename, warn_returncode=True)
        return
    f = open(filename, 'rb')
    old_content = f.read()
    f.close()
    if content == old_content:
        if self.verbose > 1:
            print('File %s matches expected content' % filename)
        return
    if not self.options.overwrite:
        print('Warning: file %s does not match expected content' % filename)
        if difflib is None:
            import difflib
        diff = difflib.context_diff(content.splitlines(), old_content.splitlines(), 'expected ' + filename, filename)
        print('\n'.join(diff))
        if self.interactive:
            while 1:
                s = input('Overwrite file with new content? [y/N] ').strip().lower()
                if not s:
                    s = 'n'
                if s.startswith('y'):
                    break
                if s.startswith('n'):
                    return
                print('Unknown response; Y or N please')
        else:
            return
    if self.verbose:
        print('Overwriting %s with new content' % filename)
    if not self.simulate:
        f = open(filename, 'wb')
        f.write(content)
        f.close()