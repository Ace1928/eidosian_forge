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
def getgroup(self, name: str, description: str='', after: Optional[str]=None) -> 'OptionGroup':
    """Get (or create) a named option Group.

        :param name: Name of the option group.
        :param description: Long description for --help output.
        :param after: Name of another group, used for ordering --help output.
        :returns: The option group.

        The returned group object has an ``addoption`` method with the same
        signature as :func:`parser.addoption <pytest.Parser.addoption>` but
        will be shown in the respective group in the output of
        ``pytest --help``.
        """
    for group in self._groups:
        if group.name == name:
            return group
    group = OptionGroup(name, description, parser=self, _ispytest=True)
    i = 0
    for i, grp in enumerate(self._groups):
        if grp.name == after:
            break
    self._groups.insert(i + 1, group)
    return group