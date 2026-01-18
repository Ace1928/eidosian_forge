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
def parse_setoption(self, args: Sequence[Union[str, 'os.PathLike[str]']], option: argparse.Namespace, namespace: Optional[argparse.Namespace]=None) -> List[str]:
    parsedoption = self.parse(args, namespace=namespace)
    for name, value in parsedoption.__dict__.items():
        setattr(option, name, value)
    return cast(List[str], getattr(parsedoption, FILE_OR_DIR))