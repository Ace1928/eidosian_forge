import logging
from collections import defaultdict
from typing import Tuple, Dict, Optional, List
import torch
from torch._export import export
from torch._export.pass_base import _ExportPassBase
from torch._export.pass_infra.node_metadata import NodeMetadata
from torch._export.pass_infra.proxy_value import ProxyValue
from torch._subclasses import FakeTensor
from torch.fx.node import Target, Argument
from torch.library import Library
from torch.utils._pytree import tree_unflatten
import torch._export.exported_program as ep
import re
def get_target_version(versioned_upgrader_name: str) -> int:
    """div_Scalar_0_3 is the name of the upgrader, meaning it applies to div.Scalar of version 0 to 3 and is
    upgrading to version 4."""
    if not re.match('^.*_[0-9]+_[0-9]+$', versioned_upgrader_name):
        raise RuntimeError(f'Upgrader name {versioned_upgrader_name} is invalid')
    return int(versioned_upgrader_name.split('_')[-1]) + 1