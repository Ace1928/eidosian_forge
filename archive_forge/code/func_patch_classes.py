import argparse
import contextlib
import functools
import types
from typing import Any, Sequence, Text, TextIO, Tuple, Type, Optional, Union
from typing import Callable, ContextManager, Generator
import autopage
from argparse import *  # noqa
def patch_classes(module: types.ModuleType, impl: Tuple[Type, ...]) -> None:
    module._HelpAction, module.HelpFormatter, module.RawDescriptionHelpFormatter, module.RawTextHelpFormatter, module.ArgumentDefaultsHelpFormatter, module.MetavarTypeHelpFormatter = impl