import fnmatch
import locale
import os
import re
import stat
import subprocess
import sys
import textwrap
import types
import warnings
from xml.etree import ElementTree
def find_file_iter(filename, env_vars=(), searchpath=(), file_names=None, url=None, verbose=False, finding_dir=False):
    """
    Search for a file to be used by nltk.

    :param filename: The name or path of the file.
    :param env_vars: A list of environment variable names to check.
    :param file_names: A list of alternative file names to check.
    :param searchpath: List of directories to search.
    :param url: URL presented to user for download help.
    :param verbose: Whether or not to print path when a file is found.
    """
    file_names = [filename] + (file_names or [])
    assert isinstance(filename, str)
    assert not isinstance(file_names, str)
    assert not isinstance(searchpath, str)
    if isinstance(env_vars, str):
        env_vars = env_vars.split()
    yielded = False
    for alternative in file_names:
        path_to_file = os.path.join(filename, alternative)
        if os.path.isfile(path_to_file):
            if verbose:
                print(f'[Found {filename}: {path_to_file}]')
            yielded = True
            yield path_to_file
        if os.path.isfile(alternative):
            if verbose:
                print(f'[Found {filename}: {alternative}]')
            yielded = True
            yield alternative
        path_to_file = os.path.join(filename, 'file', alternative)
        if os.path.isfile(path_to_file):
            if verbose:
                print(f'[Found {filename}: {path_to_file}]')
            yielded = True
            yield path_to_file
    for env_var in env_vars:
        if env_var in os.environ:
            if finding_dir:
                yielded = True
                yield os.environ[env_var]
            for env_dir in os.environ[env_var].split(os.pathsep):
                if os.path.isfile(env_dir):
                    if verbose:
                        print(f'[Found {filename}: {env_dir}]')
                    yielded = True
                    yield env_dir
                for alternative in file_names:
                    path_to_file = os.path.join(env_dir, alternative)
                    if os.path.isfile(path_to_file):
                        if verbose:
                            print(f'[Found {filename}: {path_to_file}]')
                        yielded = True
                        yield path_to_file
                    path_to_file = os.path.join(env_dir, 'bin', alternative)
                    if os.path.isfile(path_to_file):
                        if verbose:
                            print(f'[Found {filename}: {path_to_file}]')
                        yielded = True
                        yield path_to_file
    for directory in searchpath:
        for alternative in file_names:
            path_to_file = os.path.join(directory, alternative)
            if os.path.isfile(path_to_file):
                yielded = True
                yield path_to_file
    if os.name == 'posix':
        for alternative in file_names:
            try:
                p = subprocess.Popen(['which', alternative], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                stdout, stderr = p.communicate()
                path = _decode_stdoutdata(stdout).strip()
                if path.endswith(alternative) and os.path.exists(path):
                    if verbose:
                        print(f'[Found {filename}: {path}]')
                    yielded = True
                    yield path
            except (KeyboardInterrupt, SystemExit, OSError):
                raise
            finally:
                pass
    if not yielded:
        msg = 'NLTK was unable to find the %s file!\nUse software specific configuration parameters' % filename
        if env_vars:
            msg += ' or set the %s environment variable' % env_vars[0]
        msg += '.'
        if searchpath:
            msg += '\n\n  Searched in:'
            msg += ''.join(('\n    - %s' % d for d in searchpath))
        if url:
            msg += f'\n\n  For more information on {filename}, see:\n    <{url}>'
        div = '=' * 75
        raise LookupError(f'\n\n{div}\n{msg}\n{div}')