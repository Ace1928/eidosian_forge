import argparse
import functools
import os
import pathlib
import re
import sys
import textwrap
from types import ModuleType
from typing import (
from requests.structures import CaseInsensitiveDict
import gitlab.config
from gitlab.base import RESTObject
def register_custom_action(cls_names: Union[str, Tuple[str, ...]], mandatory: Tuple[str, ...]=(), optional: Tuple[str, ...]=(), custom_action: Optional[str]=None) -> Callable[[__F], __F]:

    def wrap(f: __F) -> __F:

        @functools.wraps(f)
        def wrapped_f(*args: Any, **kwargs: Any) -> Any:
            return f(*args, **kwargs)
        in_obj = True
        if isinstance(cls_names, tuple):
            classes = cls_names
        else:
            classes = (cls_names,)
        for cls_name in classes:
            final_name = cls_name
            if cls_name.endswith('Manager'):
                final_name = cls_name.replace('Manager', '')
                in_obj = False
            if final_name not in custom_actions:
                custom_actions[final_name] = {}
            action = custom_action or f.__name__.replace('_', '-')
            custom_actions[final_name][action] = (mandatory, optional, in_obj)
        return cast(__F, wrapped_f)
    return wrap