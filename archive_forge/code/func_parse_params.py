import xml.dom.minidom
import subprocess
import os
from shutil import rmtree
import keyword
from ..base import (CommandLine, CommandLineInputSpec, SEMLikeCommandLine, TraitedSpec,
import os\n\n\n"""
def parse_params(params):
    list = []
    for key, value in params.items():
        if isinstance(value, (str, bytes)):
            list.append('%s="%s"' % (key, value.replace('"', "'")))
        else:
            list.append('%s=%s' % (key, value))
    return ', '.join(list)