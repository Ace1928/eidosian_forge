from __future__ import annotations
import argparse
from flake8 import defaults
from flake8.options.manager import OptionManager
def stage1_arg_parser() -> argparse.ArgumentParser:
    """Register the preliminary options on our OptionManager.

    The preliminary options include:

    - ``-v``/``--verbose``
    - ``--output-file``
    - ``--append-config``
    - ``--config``
    - ``--isolated``
    - ``--enable-extensions``
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-v', '--verbose', default=0, action='count', help='Print more information about what is happening in flake8. This option is repeatable and will increase verbosity each time it is repeated.')
    parser.add_argument('--output-file', default=None, help='Redirect report to a file.')
    parser.add_argument('--append-config', action='append', default=[], help='Provide extra config files to parse in addition to the files found by Flake8 by default. These files are the last ones read and so they take the highest precedence when multiple files provide the same option.')
    parser.add_argument('--config', default=None, help='Path to the config file that will be the authoritative config source. This will cause Flake8 to ignore all other configuration files.')
    parser.add_argument('--isolated', default=False, action='store_true', help='Ignore all configuration files.')
    parser.add_argument('--enable-extensions', help='Enable plugins and extensions that are otherwise disabled by default')
    parser.add_argument('--require-plugins', help='Require specific plugins to be installed before running')
    return parser