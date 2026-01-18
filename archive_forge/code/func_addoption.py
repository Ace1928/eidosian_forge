import argparse
from gettext import gettext
import os
import sys
from typing import Any
from typing import Callable
from typing import cast
from typing import Dict
from typing import final
from typing import List
from typing import Literal
from typing import Mapping
from typing import NoReturn
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union
import _pytest._io
from _pytest.config.exceptions import UsageError
from _pytest.deprecated import check_ispytest
def addoption(self, *opts: str, **attrs: Any) -> None:
    """Add an option to this group.

        If a shortened version of a long option is specified, it will
        be suppressed in the help. ``addoption('--twowords', '--two-words')``
        results in help showing ``--two-words`` only, but ``--twowords`` gets
        accepted **and** the automatic destination is in ``args.twowords``.

        :param opts:
            Option names, can be short or long options.
        :param attrs:
            Same attributes as the argparse library's :meth:`add_argument()
            <argparse.ArgumentParser.add_argument>` function accepts.
        """
    conflict = set(opts).intersection((name for opt in self.options for name in opt.names()))
    if conflict:
        raise ValueError('option names %s already added' % conflict)
    option = Argument(*opts, **attrs)
    self._addoption_instance(option, shortupper=False)