import xml.dom.minidom
import subprocess
import os
from shutil import rmtree
import keyword
from ..base import (CommandLine, CommandLineInputSpec, SEMLikeCommandLine, TraitedSpec,
import os\n\n\n"""
def force_to_valid_python_variable_name(old_name):
    """Valid c++ names are not always valid in python, so
    provide alternate naming

    >>> force_to_valid_python_variable_name('lambda')
    'opt_lambda'
    >>> force_to_valid_python_variable_name('inputVolume')
    'inputVolume'
    """
    new_name = old_name
    new_name = new_name.lstrip().rstrip()
    if old_name in python_keywords:
        new_name = 'opt_' + old_name
    return new_name