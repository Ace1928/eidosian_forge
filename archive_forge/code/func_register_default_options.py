from __future__ import annotations
import argparse
from flake8 import defaults
from flake8.options.manager import OptionManager
def register_default_options(option_manager: OptionManager) -> None:
    """Register the default options on our OptionManager.

    The default options include:

    - ``-q``/``--quiet``
    - ``--color``
    - ``--count``
    - ``--exclude``
    - ``--extend-exclude``
    - ``--filename``
    - ``--format``
    - ``--hang-closing``
    - ``--ignore``
    - ``--extend-ignore``
    - ``--per-file-ignores``
    - ``--max-line-length``
    - ``--max-doc-length``
    - ``--indent-size``
    - ``--select``
    - ``--extend-select``
    - ``--disable-noqa``
    - ``--show-source``
    - ``--statistics``
    - ``--exit-zero``
    - ``-j``/``--jobs``
    - ``--tee``
    - ``--benchmark``
    - ``--bug-report``
    """
    add_option = option_manager.add_option
    add_option('-q', '--quiet', default=0, action='count', parse_from_config=True, help='Report only file names, or nothing. This option is repeatable.')
    add_option('--color', choices=('auto', 'always', 'never'), default='auto', help='Whether to use color in output.  Defaults to `%(default)s`.')
    add_option('--count', action='store_true', parse_from_config=True, help='Print total number of errors to standard output after all other output.')
    add_option('--exclude', metavar='patterns', default=','.join(defaults.EXCLUDE), comma_separated_list=True, parse_from_config=True, normalize_paths=True, help='Comma-separated list of files or directories to exclude. (Default: %(default)s)')
    add_option('--extend-exclude', metavar='patterns', default='', parse_from_config=True, comma_separated_list=True, normalize_paths=True, help='Comma-separated list of files or directories to add to the list of excluded ones.')
    add_option('--filename', metavar='patterns', default='*.py', parse_from_config=True, comma_separated_list=True, help='Only check for filenames matching the patterns in this comma-separated list. (Default: %(default)s)')
    add_option('--stdin-display-name', default='stdin', help='The name used when reporting errors from code passed via stdin. This is useful for editors piping the file contents to flake8. (Default: %(default)s)')
    add_option('--format', metavar='format', default='default', parse_from_config=True, help=f"Format errors according to the chosen formatter ({', '.join(sorted(option_manager.formatter_names))}) or a format string containing %%-style mapping keys (code, col, path, row, text). For example, ``--format=pylint`` or ``--format='%%(path)s %%(code)s'``. (Default: %(default)s)")
    add_option('--hang-closing', action='store_true', parse_from_config=True, help="Hang closing bracket instead of matching indentation of opening bracket's line.")
    add_option('--ignore', metavar='errors', parse_from_config=True, comma_separated_list=True, help=f'Comma-separated list of error codes to ignore (or skip). For example, ``--ignore=E4,E51,W234``. (Default: {','.join(defaults.IGNORE)})')
    add_option('--extend-ignore', metavar='errors', parse_from_config=True, comma_separated_list=True, help='Comma-separated list of error codes to add to the list of ignored ones. For example, ``--extend-ignore=E4,E51,W234``.')
    add_option('--per-file-ignores', default='', parse_from_config=True, help='A pairing of filenames and violation codes that defines which violations to ignore in a particular file. The filenames can be specified in a manner similar to the ``--exclude`` option and the violations work similarly to the ``--ignore`` and ``--select`` options.')
    add_option('--max-line-length', type=int, metavar='n', default=defaults.MAX_LINE_LENGTH, parse_from_config=True, help='Maximum allowed line length for the entirety of this run. (Default: %(default)s)')
    add_option('--max-doc-length', type=int, metavar='n', default=None, parse_from_config=True, help='Maximum allowed doc line length for the entirety of this run. (Default: %(default)s)')
    add_option('--indent-size', type=int, metavar='n', default=defaults.INDENT_SIZE, parse_from_config=True, help='Number of spaces used for indentation (Default: %(default)s)')
    add_option('--select', metavar='errors', parse_from_config=True, comma_separated_list=True, help='Limit the reported error codes to codes prefix-matched by this list.  You usually do not need to specify this option as the default includes all installed plugin codes.  For example, ``--select=E4,E51,W234``.')
    add_option('--extend-select', metavar='errors', parse_from_config=True, comma_separated_list=True, help='Add additional error codes to the default ``--select``.  You usually do not need to specify this option as the default includes all installed plugin codes.  For example, ``--extend-select=E4,E51,W234``.')
    add_option('--disable-noqa', default=False, parse_from_config=True, action='store_true', help='Disable the effect of "# noqa". This will report errors on lines with "# noqa" at the end.')
    add_option('--show-source', action='store_true', parse_from_config=True, help='Show the source generate each error or warning.')
    add_option('--no-show-source', action='store_false', dest='show_source', parse_from_config=False, help='Negate --show-source')
    add_option('--statistics', action='store_true', parse_from_config=True, help='Count errors.')
    add_option('--exit-zero', action='store_true', help='Exit with status code "0" even if there are errors.')
    add_option('-j', '--jobs', default='auto', parse_from_config=True, type=JobsArgument, help='Number of subprocesses to use to run checks in parallel. This is ignored on Windows. The default, "auto", will auto-detect the number of processors available to use. (Default: %(default)s)')
    add_option('--tee', default=False, parse_from_config=True, action='store_true', help='Write to stdout and output-file.')
    add_option('--benchmark', default=False, action='store_true', help='Print benchmark information about this run of Flake8')
    add_option('--bug-report', action='store_true', help='Print information necessary when preparing a bug report')