from typing import Dict, List, Optional, Sequence, Tuple
from torchgen.api import cpp
from torchgen.api.autograd import (
from torchgen.api.types import (
from torchgen.code_template import CodeTemplate
from torchgen.context import with_native_function
from torchgen.model import (
from torchgen.utils import FileManager
from .context import with_native_function_with_differentiability_info
from .gen_trace_type import (
def get_creation_meta_in_mode(original: str) -> str:
    creation_meta_with_grad_mode = f'(at::GradMode::is_enabled() ? {original} : CreationMeta::NO_GRAD_MODE)'
    return f'InferenceMode::is_enabled() ? CreationMeta::INFERENCE_MODE : {creation_meta_with_grad_mode}'