import logging
import os
import shutil
import subprocess
import sys
import sysconfig
import types
def setup_scripts(self, context):
    """
        Set up scripts into the created environment from a directory.

        This method installs the default scripts into the environment
        being created. You can prevent the default installation by overriding
        this method if you really need to, or if you need to specify
        a different location for the scripts to install. By default, the
        'scripts' directory in the venv package is used as the source of
        scripts to install.
        """
    path = os.path.abspath(os.path.dirname(__file__))
    path = os.path.join(path, 'scripts')
    self.install_scripts(context, path)