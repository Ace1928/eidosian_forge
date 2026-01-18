import argparse
import contextlib
import functools
import types
from typing import Any, Sequence, Text, TextIO, Tuple, Type, Optional, Union
from typing import Callable, ContextManager, Generator
import autopage
from argparse import *  # noqa
def get_existing_classes(module: types.ModuleType) -> Tuple[Type, ...]:
    return (module._HelpAction, module.HelpFormatter, module.RawDescriptionHelpFormatter, module.RawTextHelpFormatter, module.ArgumentDefaultsHelpFormatter, module.MetavarTypeHelpFormatter)