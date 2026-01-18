import re
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple, Union
from torchgen.api import cpp
from torchgen.api.autograd import (
from torchgen.api.types import (
from torchgen.code_template import CodeTemplate
from torchgen.context import (
from torchgen.model import (
from torchgen.utils import FileManager, mapMaybe
from .context import with_native_function_with_differentiability_info_and_key
from .gen_inplace_or_view_type import (
from .gen_trace_type import (
def emit_original_self_definition() -> List[str]:
    body: List[str] = []
    if inplace:
        if is_inplace_foreach:
            body.append('std::vector<c10::optional<at::Tensor>> original_selfs(self.size());')
        else:
            body.append('c10::optional<at::Tensor> original_self;')
        all_forward_grad_cond = []
        for derivative in fw_derivatives:
            if derivative.required_original_self_value:
                all_forward_grad_cond.append(get_any_has_forward_grad_name(derivative.var_names))
        if all_forward_grad_cond:
            if not is_inplace_foreach:
                body.append(f'if ({' || '.join(all_forward_grad_cond)}) {{')
                body.append('  original_self = self.clone();')
                body.append('}')
            else:
                current_all_forward_grad_cond = [f'{cond}[i]' for cond in all_forward_grad_cond]
                body.append('for (const auto& i : c10::irange(self.size())) {')
                body.append(f'  if ({' || '.join(current_all_forward_grad_cond)}) {{')
                body.append('    original_selfs[i] = self[i].clone();')
                body.append('  }')
                body.append('}')
    return body