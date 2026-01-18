from __future__ import absolute_import, division, print_function
import os
from ansible_collections.community.general.plugins.module_utils.cmd_runner import CmdRunner, cmd_runner_fmt
from ansible_collections.community.general.plugins.module_utils.module_helper import StateModuleHelper
@staticmethod
def split_name_and_version(package):
    """ Split the name and the version when using the NAME=VERSION syntax """
    splitted = package.split('=', 1)
    if len(splitted) == 1:
        return (splitted[0], None)
    else:
        return (splitted[0], splitted[1])