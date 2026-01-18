import os
from collections import defaultdict
from typing import Any, Dict, List, NamedTuple, Sequence, Set, Tuple
import numpy as np
from onnx import defs, helper
from onnx.backend.sample.ops import collect_sample_implementations
from onnx.backend.test.case import collect_snippets
from onnx.defs import ONNX_ML_DOMAIN, OpSchema
def display_version_link(name: str, version: int, changelog: str) -> str:
    name_with_ver = f'{name}-{version}'
    return f'<a href="{changelog}#{name_with_ver}">{version}</a>'