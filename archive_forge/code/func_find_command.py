import os
from .commands import Command
@classmethod
def find_command(cls, cmd):
    import os.path
    bzrpath = os.environ.get('BZRPATH', '')
    for dir in bzrpath.split(os.pathsep):
        if not dir:
            continue
        path = os.path.join(dir, cmd)
        if os.path.isfile(path):
            return ExternalCommand(path)
    return None