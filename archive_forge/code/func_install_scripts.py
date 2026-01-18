import logging
import os
import shutil
import subprocess
import sys
import sysconfig
import types
def install_scripts(self, context, path):
    """
        Install scripts into the created environment from a directory.

        :param context: The information for the environment creation request
                        being processed.
        :param path:    Absolute pathname of a directory containing script.
                        Scripts in the 'common' subdirectory of this directory,
                        and those in the directory named for the platform
                        being run on, are installed in the created environment.
                        Placeholder variables are replaced with environment-
                        specific values.
        """
    binpath = context.bin_path
    plen = len(path)
    for root, dirs, files in os.walk(path):
        if root == path:
            for d in dirs[:]:
                if d not in ('common', os.name):
                    dirs.remove(d)
            continue
        for f in files:
            if os.name == 'nt' and f.startswith('python') and f.endswith(('.exe', '.pdb')):
                continue
            srcfile = os.path.join(root, f)
            suffix = root[plen:].split(os.sep)[2:]
            if not suffix:
                dstdir = binpath
            else:
                dstdir = os.path.join(binpath, *suffix)
            if not os.path.exists(dstdir):
                os.makedirs(dstdir)
            dstfile = os.path.join(dstdir, f)
            with open(srcfile, 'rb') as f:
                data = f.read()
            if not srcfile.endswith(('.exe', '.pdb')):
                try:
                    data = data.decode('utf-8')
                    data = self.replace_variables(data, context)
                    data = data.encode('utf-8')
                except UnicodeError as e:
                    data = None
                    logger.warning('unable to copy script %r, may be binary: %s', srcfile, e)
            if data is not None:
                with open(dstfile, 'wb') as f:
                    f.write(data)
                shutil.copymode(srcfile, dstfile)