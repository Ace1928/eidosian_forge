import keyword
import warnings
import weakref
from collections import OrderedDict, defaultdict, deque
from copy import deepcopy
from itertools import islice, zip_longest
from types import BuiltinFunctionType, CodeType, FunctionType, GeneratorType, LambdaType, ModuleType
from typing import (
from typing_extensions import Annotated
from .errors import ConfigError
from .typing import (
from .version import version_info
def get_unique_discriminator_alias(all_aliases: Collection[str], discriminator_key: str) -> str:
    """Validate that all aliases are the same and if that's the case return the alias"""
    unique_aliases = set(all_aliases)
    if len(unique_aliases) > 1:
        raise ConfigError(f'Aliases for discriminator {discriminator_key!r} must be the same (got {', '.join(sorted(all_aliases))})')
    return unique_aliases.pop()