from __future__ import annotations
import os
import shutil
import subprocess
import sys
from shlex import quote
from . import Image
def get_command_ex(self, file, title=None, **options):
    command = executable = 'xv'
    if title:
        command += f' -name {quote(title)}'
    return (command, executable)