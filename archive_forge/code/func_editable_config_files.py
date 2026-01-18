import types
import os
import string
import uuid
from paste.deploy import appconfig
from paste.script import copydir
from paste.script.command import Command, BadCommand, run as run_command
from paste.script.util import secret
from paste.util import import_string
import paste.script.templates
import pkg_resources
def editable_config_files(self, filename):
    """
        Return a list of filenames; this is primarily used when the
        filename is treated as a directory and several configuration
        files are created.  The default implementation returns the
        file itself.  Return None if you don't know what files should
        be edited on installation.
        """
    if not self.expect_config_directory:
        return [filename]
    else:
        return None