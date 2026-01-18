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
def simplify(feature: dict) -> dict:
    if not isinstance(feature, dict):
        raise TypeError(f'Expected a dict but got a {type(feature)}: {feature}')
    if isinstance(feature.get('sequence'), dict) and list(feature['sequence']) == ['dtype']:
        feature['sequence'] = feature['sequence']['dtype']
    if isinstance(feature.get('sequence'), dict) and list(feature['sequence']) == ['struct']:
        feature['sequence'] = feature['sequence']['struct']
    if isinstance(feature.get('list'), dict) and list(feature['list']) == ['dtype']:
        feature['list'] = feature['list']['dtype']
    if isinstance(feature.get('list'), dict) and list(feature['list']) == ['struct']:
        feature['list'] = feature['list']['struct']
    if isinstance(feature.get('class_label'), dict) and isinstance(feature['class_label'].get('names'), list):
        feature['class_label']['names'] = {str(label_id): label_name for label_id, label_name in enumerate(feature['class_label']['names'])}
    return feature