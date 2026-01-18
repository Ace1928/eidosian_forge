import itertools
import re
from collections import defaultdict
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple
import yaml
from torchgen.api import cpp
from torchgen.api.python import (
from torchgen.code_template import CodeTemplate
from torchgen.context import with_native_function
from torchgen.gen import cpp_string, parse_native_yaml, parse_tags_yaml
from torchgen.model import (
from torchgen.utils import FileManager, split_name_params
from torchgen.yaml_utils import YamlLoader
from .gen_trace_type import should_trace
def is_arg_smaller(t1: Type, t2: Type) -> bool:
    return str(t1) == 'Scalar' and str(t2) == 'Tensor' or (str(t1) == 'Scalar?' and str(t2) == 'Tensor?') or ('Dimname' in str(t1) and 'Dimname' not in str(t2)) or (str(t1) == 'int[]' and (str(t2) == 'int' or str(t2) == 'int?')) or (str(t1) == 'Tensor[]' and str(t2).find('[]') != -1) or (str(t1) == 'SymInt[]' and str(t2) == 'int[]') or ((str(t1) == 'SymInt' or str(t1) == 'int') and str(t2) == 'Tensor')