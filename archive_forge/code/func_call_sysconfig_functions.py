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
def call_sysconfig_functions(self, name, *args, **kw):
    """
        Call all the named functions in the sysconfig modules,
        returning a list of the return values.
        """
    return [method(*args, **kw) for method in self.get_sysconfig_options(name)]