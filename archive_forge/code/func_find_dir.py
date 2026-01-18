import os
import pkg_resources
from paste.script import pluginlib, copydir
from paste.script.command import BadCommand
import subprocess
def find_dir(self, dirname, package=False):
    egg_info = pluginlib.find_egg_info_dir(os.getcwd())
    f = open(os.path.join(egg_info, 'top_level.txt'))
    packages = [l.strip() for l in f.readlines() if l.strip() and (not l.strip().startswith('#'))]
    f.close()
    if not len(packages):
        raise BadCommand('No top level dir found for %s' % dirname)
    base = os.path.dirname(egg_info)
    possible = []
    for pkg in packages:
        d = os.path.join(base, pkg, dirname)
        if os.path.exists(d):
            possible.append((pkg, d))
    if not possible:
        self.ensure_dir(os.path.join(base, packages[0], dirname), package=package)
        return self.find_dir(dirname)
    if len(possible) > 1:
        raise BadCommand('Multiple %s dirs found (%s)' % (dirname, possible))
    return possible[0]