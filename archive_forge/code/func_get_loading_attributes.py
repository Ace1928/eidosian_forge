import copy
import importlib.metadata
import json
import os
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from packaging import version
from ..utils import is_auto_awq_available, is_torch_available, logging
def get_loading_attributes(self):
    attibutes_dict = copy.deepcopy(self.__dict__)
    loading_attibutes = ['do_fuse', 'modules_to_fuse', 'fuse_max_seq_len']
    loading_attibutes_dict = {i: j for i, j in attibutes_dict.items() if i in loading_attibutes}
    return loading_attibutes_dict