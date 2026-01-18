import re
import shlex
from typing import (
import attr
from . import (
from .exceptions import (
@property
def argv(self) -> List[str]:
    """a list of arguments a-la ``sys.argv``.

        The first element of the list is the command after shortcut and macro
        expansion. Subsequent elements of the list contain any additional
        arguments, with quotes removed, just like bash would. This is very
        useful if you are going to use ``argparse.parse_args()``.

        If you want to strip quotes from the input, you can use ``argv[1:]``.
        """
    if self.command:
        rtn = [utils.strip_quotes(self.command)]
        for cur_token in self.arg_list:
            rtn.append(utils.strip_quotes(cur_token))
    else:
        rtn = []
    return rtn