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
def config_content(self, command, vars):
    """
        Called by ``self.write_config``, this returns the text content
        for the config file, given the provided variables.

        The default implementation reads
        ``Package.egg-info/paste_deploy_config.ini_tmpl`` and fills it
        with the variables.
        """
    global Cheetah
    meta_name = 'paste_deploy_config.ini_tmpl'
    if not self.dist.has_metadata(meta_name):
        if command.verbose:
            print('No %s found' % meta_name)
        return self.simple_config(vars)
    return self.template_renderer(self.dist.get_metadata(meta_name), vars, filename=meta_name)