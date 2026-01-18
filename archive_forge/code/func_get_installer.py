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
def get_installer(self, distro, ep_group, ep_name):
    if hasattr(distro, 'load_entry_point'):
        installer_class = distro.load_entry_point('paste.app_install', ep_name)
    else:
        eps = [ep for ep in distro.entry_points if ep.group == 'paste.app_install' and ep.name == ep_name]
        installer_class = eps[0].load()
    installer = installer_class(distro, ep_group, ep_name)
    return installer