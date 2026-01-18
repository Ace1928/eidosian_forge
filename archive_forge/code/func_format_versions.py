import os
from collections import defaultdict
from typing import Any, Dict, List, NamedTuple, Sequence, Set, Tuple
import numpy as np
from onnx import defs, helper
from onnx.backend.sample.ops import collect_sample_implementations
from onnx.backend.test.case import collect_snippets
from onnx.defs import ONNX_ML_DOMAIN, OpSchema
def format_versions(versions: Sequence[OpSchema], changelog: str) -> str:
    return f'{', '.join((display_version_link(format_name_with_domain(v.domain, v.name), v.since_version, changelog) for v in versions[::-1]))}'