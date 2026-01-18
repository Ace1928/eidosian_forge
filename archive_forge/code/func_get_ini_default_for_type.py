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
def get_ini_default_for_type(type: Optional[Literal['string', 'paths', 'pathlist', 'args', 'linelist', 'bool']]) -> Any:
    """
    Used by addini to get the default value for a given ini-option type, when
    default is not supplied.
    """
    if type is None:
        return ''
    elif type in ('paths', 'pathlist', 'args', 'linelist'):
        return []
    elif type == 'bool':
        return False
    else:
        return ''