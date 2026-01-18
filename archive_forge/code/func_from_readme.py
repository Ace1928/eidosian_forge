import re
import textwrap
from collections import Counter
from itertools import groupby
from operator import itemgetter
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Union
import yaml
from huggingface_hub import DatasetCardData
from ..config import METADATA_CONFIGS_FIELD
from ..info import DatasetInfo, DatasetInfosDict
from ..naming import _split_re
from ..utils.logging import get_logger
from .deprecation_utils import deprecated
@classmethod
def from_readme(cls, path: Union[Path, str]) -> 'DatasetMetadata':
    """Loads and validates the dataset metadata from its dataset card (README.md)

        Args:
            path (:obj:`Path`): Path to the dataset card (its README.md file)

        Returns:
            :class:`DatasetMetadata`: The dataset's metadata

        Raises:
            :obj:`TypeError`: If the dataset's metadata is invalid
        """
    with open(path, encoding='utf-8') as readme_file:
        yaml_string, _ = _split_yaml_from_readme(readme_file.read())
    if yaml_string is not None:
        return cls.from_yaml_string(yaml_string)
    else:
        return cls()