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
def template_renderer(self, content, vars, filename=None):
    """
        Subclasses may override this to provide different template
        substitution (e.g., use a different template engine).
        """
    if self.use_cheetah:
        import Cheetah.Template
        tmpl = Cheetah.Template.Template(content, searchList=[vars])
        return copydir.careful_sub(tmpl, vars, filename)
    else:
        tmpl = string.Template(content)
        return tmpl.substitute(vars)