import fnmatch
import os
import re
import sys
import pkg_resources
from . import copydir
from . import pluginlib
from .command import Command, BadCommand
def setup_svn_repository(self, output_dir, dist_name):
    svn_repos = self.options.svn_repository
    svn_repos_path = os.path.join(svn_repos, dist_name).replace('\\', '/')
    svn_command = 'svn'
    if sys.platform == 'win32':
        svn_command += '.exe'
    cmd = '%(svn_command)s mkdir %(svn_repos_path)s' + ' %(svn_repos_path)s/trunk %(svn_repos_path)s/tags' + ' %(svn_repos_path)s/branches -m "New project %(dist_name)s"'
    cmd = cmd % {'svn_repos_path': svn_repos_path, 'dist_name': dist_name, 'svn_command': svn_command}
    if self.verbose:
        print('Running:')
        print(cmd)
    if not self.simulate:
        os.system(cmd)
    svn_repos_path_trunk = os.path.join(svn_repos_path, 'trunk').replace('\\', '/')
    cmd = svn_command + ' co "%s" "%s"' % (svn_repos_path_trunk, output_dir)
    if self.verbose:
        print('Running %s' % cmd)
    if not self.simulate:
        os.system(cmd)