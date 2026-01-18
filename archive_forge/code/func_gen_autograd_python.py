import argparse
import os
from typing import List
from torchgen.api import cpp
from torchgen.api.autograd import (
from torchgen.gen import parse_native_yaml
from torchgen.selective_build.selector import SelectiveBuilder
from . import gen_python_functions
from .gen_autograd_functions import (
from .gen_inplace_or_view_type import gen_inplace_or_view_type
from .gen_trace_type import gen_trace_type
from .gen_variable_factories import gen_variable_factories
from .gen_variable_type import gen_variable_type
from .load_derivatives import load_derivatives
def gen_autograd_python(native_functions_path: str, tags_path: str, out: str, autograd_dir: str) -> None:
    differentiability_infos, _ = load_derivatives(os.path.join(autograd_dir, 'derivatives.yaml'), native_functions_path, tags_path)
    template_path = os.path.join(autograd_dir, 'templates')
    gen_autograd_functions_python(out, differentiability_infos, template_path)
    deprecated_path = os.path.join(autograd_dir, 'deprecated.yaml')
    gen_python_functions.gen(out, native_functions_path, tags_path, deprecated_path, template_path)