import os
import pkg_resources
from paste.script import pluginlib, copydir
from paste.script.command import BadCommand
import subprocess
def parse_path_name_args(self, name):
    """
        Given the name, assume that the first argument is a path/filename
        combination. Return the name and dir of this. If the name ends with
        '.py' that will be erased.

        Examples:
            comments             ->          comments, ''
            admin/comments       ->          comments, 'admin'
            h/ab/fred            ->          fred, 'h/ab'
        """
    if name.endswith('.py'):
        name = name[:-3]
    if '.' in name:
        name = name.replace('.', os.path.sep)
    if '/' != os.path.sep:
        name = name.replace('/', os.path.sep)
    parts = name.split(os.path.sep)
    name = parts[-1]
    if not parts[:-1]:
        dir = ''
    elif len(parts[:-1]) == 1:
        dir = parts[0]
    else:
        dir = os.path.join(*parts[:-1])
    return (name, dir)