import abc
import collections
import copy
import dataclasses
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
from .arrow_reader import FileInstructions, make_file_instructions
from .naming import _split_re
from .utils.py_utils import NonMutableDict, asdict
@classmethod
def from_split_dict(cls, split_infos: Union[List, Dict], dataset_name: Optional[str]=None):
    """Returns a new SplitDict initialized from a Dict or List of `split_infos`."""
    if isinstance(split_infos, dict):
        split_infos = list(split_infos.values())
    if dataset_name is None:
        dataset_name = split_infos[0].get('dataset_name') if split_infos else None
    split_dict = cls(dataset_name=dataset_name)
    for split_info in split_infos:
        if isinstance(split_info, dict):
            split_info = SplitInfo(**split_info)
        split_dict.add(split_info)
    return split_dict