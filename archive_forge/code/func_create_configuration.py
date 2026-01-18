import logging
import os
import shutil
import subprocess
import sys
import sysconfig
import types
def create_configuration(self, context):
    """
        Create a configuration file indicating where the environment's Python
        was copied from, and whether the system site-packages should be made
        available in the environment.

        :param context: The information for the environment creation request
                        being processed.
        """
    context.cfg_path = path = os.path.join(context.env_dir, 'pyvenv.cfg')
    with open(path, 'w', encoding='utf-8') as f:
        f.write('home = %s\n' % context.python_dir)
        if self.system_site_packages:
            incl = 'true'
        else:
            incl = 'false'
        f.write('include-system-site-packages = %s\n' % incl)
        f.write('version = %d.%d.%d\n' % sys.version_info[:3])
        if self.prompt is not None:
            f.write(f'prompt = {self.prompt!r}\n')
        f.write('executable = %s\n' % os.path.realpath(sys.executable))
        args = []
        nt = os.name == 'nt'
        if nt and self.symlinks:
            args.append('--symlinks')
        if not nt and (not self.symlinks):
            args.append('--copies')
        if not self.with_pip:
            args.append('--without-pip')
        if self.system_site_packages:
            args.append('--system-site-packages')
        if self.clear:
            args.append('--clear')
        if self.upgrade:
            args.append('--upgrade')
        if self.upgrade_deps:
            args.append('--upgrade-deps')
        if self.orig_prompt is not None:
            args.append(f'--prompt="{self.orig_prompt}"')
        args.append(context.env_dir)
        args = ' '.join(args)
        f.write(f'command = {sys.executable} -m venv {args}\n')