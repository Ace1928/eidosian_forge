from functools import wraps, partial
from itertools import product, chain, islice
import itertools
import functools
import copy
import operator
import random
import unittest
import math
import enum
import torch
import numpy as np
from torch import inf, nan
from typing import Any, Dict, List, Tuple, Union, Sequence
from torch.testing import make_tensor
from torch.testing._internal.common_dtype import (
from torch.testing._internal.common_device_type import \
from torch.testing._internal.common_cuda import (
from torch.testing._internal.common_utils import (
import torch._refs as refs  # noqa: F401
import torch._refs.nn.functional
import torch._refs.special
import torch._refs.linalg
import torch._prims as prims  # noqa: F401
from torch.utils import _pytree as pytree
from packaging import version
from torch.testing._internal.opinfo.core import (  # noqa: F401
from torch.testing._internal.opinfo.refs import (  # NOQA: F401
from torch.testing._internal.opinfo.utils import (
from torch.testing._internal import opinfo
from torch.testing._internal.opinfo.definitions.linalg import (
from torch.testing._internal.opinfo.definitions.special import (
from torch.testing._internal.opinfo.definitions._masked import (
from torch.testing._internal.opinfo.definitions.sparse import (
def make_mvlgamma_opinfo(variant_test_name, domain, skips, sample_kwargs):
    return UnaryUfuncInfo('mvlgamma', ref=reference_mvlgamma if TEST_SCIPY else None, aliases=('special.multigammaln',), variant_test_name=variant_test_name, domain=domain, decorators=(precisionOverride({torch.float16: 0.05}),), dtypes=all_types_and(torch.half, torch.bfloat16), dtypesIfCUDA=all_types_and(torch.float16), sample_inputs_func=sample_inputs_mvlgamma, supports_forward_ad=True, supports_fwgrad_bwgrad=True, promotes_int_to_float=True, skips=skips, sample_kwargs=sample_kwargs)