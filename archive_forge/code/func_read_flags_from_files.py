from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import copy
import itertools
import logging
import os
import sys
from xml.dom import minidom
from absl.flags import _exceptions
from absl.flags import _flag
from absl.flags import _helpers
from absl.flags import _validators_classes
import six
def read_flags_from_files(self, argv, force_gnu=True):
    """Processes command line args, but also allow args to be read from file.

    Args:
      argv: [str], a list of strings, usually sys.argv[1:], which may contain
        one or more flagfile directives of the form --flagfile="./filename".
        Note that the name of the program (sys.argv[0]) should be omitted.
      force_gnu: bool, if False, --flagfile parsing obeys the
        FLAGS.is_gnu_getopt() value. If True, ignore the value and always follow
        gnu_getopt semantics.

    Returns:
      A new list which has the original list combined with what we read
      from any flagfile(s).

    Raises:
      IllegalFlagValueError: Raised when --flagfile is provided with no
          argument.

    This function is called by FLAGS(argv).
    It scans the input list for a flag that looks like:
    --flagfile=<somefile>. Then it opens <somefile>, reads all valid key
    and value pairs and inserts them into the input list in exactly the
    place where the --flagfile arg is found.

    Note that your application's flags are still defined the usual way
    using absl.flags DEFINE_flag() type functions.

    Notes (assuming we're getting a commandline of some sort as our input):
    --> For duplicate flags, the last one we hit should "win".
    --> Since flags that appear later win, a flagfile's settings can be "weak"
        if the --flagfile comes at the beginning of the argument sequence,
        and it can be "strong" if the --flagfile comes at the end.
    --> A further "--flagfile=<otherfile.cfg>" CAN be nested in a flagfile.
        It will be expanded in exactly the spot where it is found.
    --> In a flagfile, a line beginning with # or // is a comment.
    --> Entirely blank lines _should_ be ignored.
    """
    rest_of_args = argv
    new_argv = []
    while rest_of_args:
        current_arg = rest_of_args[0]
        rest_of_args = rest_of_args[1:]
        if self._is_flag_file_directive(current_arg):
            if current_arg == '--flagfile' or current_arg == '-flagfile':
                if not rest_of_args:
                    raise _exceptions.IllegalFlagValueError('--flagfile with no argument')
                flag_filename = os.path.expanduser(rest_of_args[0])
                rest_of_args = rest_of_args[1:]
            else:
                flag_filename = self._extract_filename(current_arg)
            new_argv.extend(self._get_flag_file_lines(flag_filename))
        else:
            new_argv.append(current_arg)
            if current_arg == '--':
                break
            if not current_arg.startswith('-'):
                if not force_gnu and (not self.__dict__['__use_gnu_getopt']):
                    break
            elif '=' not in current_arg and rest_of_args and (not rest_of_args[0].startswith('-')):
                fl = self._flags()
                name = current_arg.lstrip('-')
                if name in fl and (not fl[name].boolean):
                    current_arg = rest_of_args[0]
                    rest_of_args = rest_of_args[1:]
                    new_argv.append(current_arg)
    if rest_of_args:
        new_argv.extend(rest_of_args)
    return new_argv