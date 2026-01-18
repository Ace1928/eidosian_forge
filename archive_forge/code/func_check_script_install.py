import glob
import os
import sys
import tarfile
import fixtures
from pbr.tests import base
def check_script_install(self, install_stdout):
    for cmd_name in self.cmd_names:
        install_txt = 'Installing %s script to %s' % (cmd_name, self.temp_dir)
        self.assertIn(install_txt, install_stdout)
        cmd_filename = os.path.join(self.temp_dir, cmd_name)
        script_txt = open(cmd_filename, 'r').read()
        self.assertNotIn('pkg_resources', script_txt)
        stdout, _, return_code = self._run_cmd(cmd_filename)
        self.assertIn('PBR', stdout)