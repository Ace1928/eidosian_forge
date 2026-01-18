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
def get_sysconfig_options(self, name):
    """
        Return the option value for the given name in all the
        sysconfig modules in which is is found (``[]`` if none).
        """
    return [getattr(mod, name) for mod in self.sysconfig_modules if hasattr(mod, name)]