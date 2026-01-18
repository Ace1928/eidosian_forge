import re
from collections import defaultdict
from typing import Any, Counter, Dict, List, Match, Optional, Sequence, Set, Tuple
import yaml
from torchgen.api import cpp
from torchgen.api.autograd import (
from torchgen.api.types import (
from torchgen.context import with_native_function
from torchgen.gen import get_grouped_by_view_native_functions, parse_native_yaml
from torchgen.model import (
from torchgen.utils import concatMap, IDENT_REGEX, split_name_params
from torchgen.yaml_utils import YamlLoader
def check_grad_usage(defn_name: str, derivatives: Sequence[Derivative]) -> None:
    """
        Check for some subtle mistakes one might make when writing derivatives.
        These mistakes will compile, but will be latent until a function is
        used with double backwards.
        """
    uses_grad = False
    num_grads_uses = 0
    uses_named_grads = False
    used_grads_indices: List[int] = []
    for d in derivatives:
        formula = d.formula
        uses_grad = uses_grad or bool(re.findall(IDENT_REGEX.format('grad'), formula))
        num_grads_uses += len(re.findall(IDENT_REGEX.format('grads'), formula))
        uses_named_grads = uses_named_grads or bool(d.named_gradients)
        used_grads_indices.extend(used_gradient_indices(formula))
    assert num_grads_uses >= len(used_grads_indices)
    only_used_grads_indices = num_grads_uses == len(used_grads_indices)
    if uses_grad and num_grads_uses > 0:
        raise RuntimeError(f"Derivative definition of {defn_name} in derivatives.yaml illegally mixes use of 'grad' and 'grads'. Consider replacing occurrences of 'grad' with 'grads[0]'")
    if only_used_grads_indices and set(used_grads_indices) == {0}:
        raise RuntimeError(f"Derivative definition of {defn_name} in derivatives.yaml solely refers to 'grads[0]'.  If the first output is indeed the only differentiable output, replace 'grads[0]' with 'grad'; otherwise, there is a likely error in your derivatives declaration.")
    if uses_named_grads and (uses_grad or num_grads_uses > 0):
        raise RuntimeError(f'Derivative definition of {defn_name} in derivatives.yaml illegally mixes use of "grad_RETURN_NAME" and "grad" or "grads[x]". Use only one method for identifying gradients.')