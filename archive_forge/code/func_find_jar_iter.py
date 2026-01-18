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
def find_jar_iter(name_pattern, path_to_jar=None, env_vars=(), searchpath=(), url=None, verbose=False, is_regex=False):
    """
    Search for a jar that is used by nltk.

    :param name_pattern: The name of the jar file
    :param path_to_jar: The user-supplied jar location, or None.
    :param env_vars: A list of environment variable names to check
                     in addition to the CLASSPATH variable which is
                     checked by default.
    :param searchpath: List of directories to search.
    :param is_regex: Whether name is a regular expression.
    """
    assert isinstance(name_pattern, str)
    assert not isinstance(searchpath, str)
    if isinstance(env_vars, str):
        env_vars = env_vars.split()
    yielded = False
    env_vars = ['CLASSPATH'] + list(env_vars)
    if path_to_jar is not None:
        if os.path.isfile(path_to_jar):
            yielded = True
            yield path_to_jar
        else:
            raise LookupError(f'Could not find {name_pattern} jar file at {path_to_jar}')
    for env_var in env_vars:
        if env_var in os.environ:
            if env_var == 'CLASSPATH':
                classpath = os.environ['CLASSPATH']
                for cp in classpath.split(os.path.pathsep):
                    cp = os.path.expanduser(cp)
                    if os.path.isfile(cp):
                        filename = os.path.basename(cp)
                        if is_regex and re.match(name_pattern, filename) or (not is_regex and filename == name_pattern):
                            if verbose:
                                print(f'[Found {name_pattern}: {cp}]')
                            yielded = True
                            yield cp
                    if os.path.isdir(cp):
                        if not is_regex:
                            if os.path.isfile(os.path.join(cp, name_pattern)):
                                if verbose:
                                    print(f'[Found {name_pattern}: {cp}]')
                                yielded = True
                                yield os.path.join(cp, name_pattern)
                        else:
                            for file_name in os.listdir(cp):
                                if re.match(name_pattern, file_name):
                                    if verbose:
                                        print('[Found %s: %s]' % (name_pattern, os.path.join(cp, file_name)))
                                    yielded = True
                                    yield os.path.join(cp, file_name)
            else:
                jar_env = os.path.expanduser(os.environ[env_var])
                jar_iter = (os.path.join(jar_env, path_to_jar) for path_to_jar in os.listdir(jar_env)) if os.path.isdir(jar_env) else (jar_env,)
                for path_to_jar in jar_iter:
                    if os.path.isfile(path_to_jar):
                        filename = os.path.basename(path_to_jar)
                        if is_regex and re.match(name_pattern, filename) or (not is_regex and filename == name_pattern):
                            if verbose:
                                print(f'[Found {name_pattern}: {path_to_jar}]')
                            yielded = True
                            yield path_to_jar
    for directory in searchpath:
        if is_regex:
            for filename in os.listdir(directory):
                path_to_jar = os.path.join(directory, filename)
                if os.path.isfile(path_to_jar):
                    if re.match(name_pattern, filename):
                        if verbose:
                            print(f'[Found {filename}: {path_to_jar}]')
                yielded = True
                yield path_to_jar
        else:
            path_to_jar = os.path.join(directory, name_pattern)
            if os.path.isfile(path_to_jar):
                if verbose:
                    print(f'[Found {name_pattern}: {path_to_jar}]')
                yielded = True
                yield path_to_jar
    if not yielded:
        msg = 'NLTK was unable to find %s!' % name_pattern
        if env_vars:
            msg += ' Set the %s environment variable' % env_vars[0]
        msg = textwrap.fill(msg + '.', initial_indent='  ', subsequent_indent='  ')
        if searchpath:
            msg += '\n\n  Searched in:'
            msg += ''.join(('\n    - %s' % d for d in searchpath))
        if url:
            msg += '\n\n  For more information, on {}, see:\n    <{}>'.format(name_pattern, url)
        div = '=' * 75
        raise LookupError(f'\n\n{div}\n{msg}\n{div}')