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
def emit_any_has_forward_grad() -> List[str]:
    content: List[str] = []
    if not is_foreach:
        for derivative in fw_derivatives:
            requires_fw_grad = get_any_has_fw_grad_cond(derivative=derivative)
            if info and info.output_differentiability_conditions:
                assert len(info.output_differentiability_conditions) == 1
                requires_fw_grad = f'({info.output_differentiability_conditions[0]}) && {requires_fw_grad}'
            content.append(f'[[maybe_unused]] auto {get_any_has_forward_grad_name(derivative.var_names)} = {requires_fw_grad};')
    else:
        for derivative in fw_derivatives:
            bool_vector_name = get_any_has_forward_grad_name(derivative.var_names)
            cur_derivative_conditions = []
            for inp in differentiable_inputs:
                if derivative.required_inputs_fw_grad is None:
                    continue
                if inp.name not in derivative.required_inputs_fw_grad:
                    continue
                inp_name = inp.name if not inplace else refargname2inplace_foreacharg[inp.name].name
                inp_type = inp.type if not inplace else refargname2inplace_foreacharg[inp.name].type
                is_list_type = is_tensor_list_type(inp_type)
                if is_list_type:
                    if inp_name != 'self':
                        content.append(FW_DERIVATIVE_SIZE_CHECK_TEMPLATE.substitute(inp_name=inp_name))
                    cur_derivative_conditions.append(FW_DERIVATIVE_CHECK_TEMPLATE.substitute(req_inp=inp_name + '[i]'))
                else:
                    cur_derivative_conditions.append(FW_DERIVATIVE_CHECK_TEMPLATE.substitute(req_inp=inp_name))
            content.append(f'std::vector<bool> {bool_vector_name}(self.size());')
            content.append('for (const auto& i : c10::irange(self.size())) {')
            content.append(f'  {bool_vector_name}[i] = {' || '.join(cur_derivative_conditions)};')
            content.append('}')
    return content