import copy
import json
import re
import sys
from collections.abc import Iterable, Mapping
from collections.abc import Sequence as SequenceABC
from dataclasses import InitVar, dataclass, field, fields
from functools import reduce, wraps
from operator import mul
from typing import Any, Callable, ClassVar, Dict, List, Optional, Tuple, Union
from typing import Sequence as Sequence_
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.types
import pyarrow_hotfix  # noqa: F401  # to fix vulnerability on pyarrow<14.0.1
from pandas.api.extensions import ExtensionArray as PandasExtensionArray
from pandas.api.extensions import ExtensionDtype as PandasExtensionDtype
from .. import config
from ..naming import camelcase_to_snakecase, snakecase_to_camelcase
from ..table import array_cast
from ..utils import logging
from ..utils.py_utils import asdict, first_non_null_value, zip_dict
from .audio import Audio
from .image import Image, encode_pil_image
from .translation import Translation, TranslationVariableLanguages
def recursive_reorder(source, target, stack=''):
    stack_position = ' at ' + stack[1:] if stack else ''
    if isinstance(target, Sequence):
        target = target.feature
        if isinstance(target, dict):
            target = {k: [v] for k, v in target.items()}
        else:
            target = [target]
    if isinstance(source, Sequence):
        source, id_, length = (source.feature, source.id, source.length)
        if isinstance(source, dict):
            source = {k: [v] for k, v in source.items()}
            reordered = recursive_reorder(source, target, stack)
            return Sequence({k: v[0] for k, v in reordered.items()}, id=id_, length=length)
        else:
            source = [source]
            reordered = recursive_reorder(source, target, stack)
            return Sequence(reordered[0], id=id_, length=length)
    elif isinstance(source, dict):
        if not isinstance(target, dict):
            raise ValueError(f'Type mismatch: between {source} and {target}' + stack_position)
        if sorted(source) != sorted(target):
            message = f'Keys mismatch: between {source} (source) and {target} (target).\n{source.keys() - target.keys()} are missing from target and {target.keys() - source.keys()} are missing from source' + stack_position
            raise ValueError(message)
        return {key: recursive_reorder(source[key], target[key], stack + f'.{key}') for key in target}
    elif isinstance(source, list):
        if not isinstance(target, list):
            raise ValueError(f'Type mismatch: between {source} and {target}' + stack_position)
        if len(source) != len(target):
            raise ValueError(f'Length mismatch: between {source} and {target}' + stack_position)
        return [recursive_reorder(source[i], target[i], stack + '.<list>') for i in range(len(target))]
    else:
        return source