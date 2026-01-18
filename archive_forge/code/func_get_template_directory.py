from __future__ import annotations
from argparse import ArgumentParser
from argparse import Namespace
from configparser import ConfigParser
import inspect
import os
import sys
from typing import Any
from typing import cast
from typing import Dict
from typing import Mapping
from typing import Optional
from typing import overload
from typing import Sequence
from typing import TextIO
from typing import Union
from typing_extensions import TypedDict
from . import __version__
from . import command
from . import util
from .util import compat
def get_template_directory(self) -> str:
    """Return the directory where Alembic setup templates are found.

        This method is used by the alembic ``init`` and ``list_templates``
        commands.

        """
    import alembic
    package_dir = os.path.abspath(os.path.dirname(alembic.__file__))
    return os.path.join(package_dir, 'templates')