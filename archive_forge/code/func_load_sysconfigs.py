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
def load_sysconfigs(self):
    configs = self.sysconfigs[:]
    configs.reverse()
    self.sysconfig_modules = []
    for index, (explicit, name) in enumerate(configs):
        if name.endswith('.py'):
            if not os.path.exists(name):
                if explicit:
                    raise BadCommand('sysconfig file %s does not exist' % name)
                else:
                    continue
            globs = {}
            exec(compile(open(name).read(), name, 'exec'), globs)
            mod = types.ModuleType('__sysconfig_%i__' % index)
            for name, value in globs.items():
                setattr(mod, name, value)
            mod.__file__ = name
        else:
            try:
                mod = import_string.simple_import(name)
            except ImportError:
                if explicit:
                    raise
                else:
                    continue
        mod.paste_command = self
        self.sysconfig_modules.insert(0, mod)
    parser = self.parser
    self.call_sysconfig_functions('add_custom_options', parser)