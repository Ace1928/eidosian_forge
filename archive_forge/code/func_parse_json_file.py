import dataclasses
import json
import sys
import types
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, ArgumentTypeError
from copy import copy
from enum import Enum
from inspect import isclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Literal, NewType, Optional, Tuple, Union, get_type_hints
import yaml
def parse_json_file(self, json_file: str, allow_extra_keys: bool=False) -> Tuple[DataClass, ...]:
    """
        Alternative helper method that does not use `argparse` at all, instead loading a json file and populating the
        dataclass types.

        Args:
            json_file (`str` or `os.PathLike`):
                File name of the json file to parse
            allow_extra_keys (`bool`, *optional*, defaults to `False`):
                Defaults to False. If False, will raise an exception if the json file contains keys that are not
                parsed.

        Returns:
            Tuple consisting of:

                - the dataclass instances in the same order as they were passed to the initializer.
        """
    with open(Path(json_file), encoding='utf-8') as open_json_file:
        data = json.loads(open_json_file.read())
    outputs = self.parse_dict(data, allow_extra_keys=allow_extra_keys)
    return tuple(outputs)