from typing import Dict, List, Sequence, Tuple
from torchgen.api.autograd import (
from torchgen.api.types import (
from torchgen.code_template import CodeTemplate
from torchgen.model import Argument, FunctionSchema
from torchgen.utils import FileManager
from .gen_inplace_or_view_type import VIEW_FUNCTIONS
def gen_autograd_functions_lib(out: str, differentiability_infos: Dict[FunctionSchema, Dict[str, DifferentiabilityInfo]], template_path: str) -> None:
    """Functions.h and Functions.cpp body

    These contain the auto-generated subclasses of torch::autograd::Node
    for each every differentiable torch function.
    """
    infos = get_infos_with_derivatives_list(differentiability_infos)
    declarations = [process_function(f, FUNCTION_DECLARATION) for f in infos]
    definitions = [process_function(f, FUNCTION_DEFINITION) for f in infos]
    file_basename = 'Functions'
    fm = FileManager(install_dir=out, template_dir=template_path, dry_run=False)
    for suffix in ['.h', '.cpp']:
        fname = file_basename + suffix
        fm.write_with_template(fname, fname, lambda: {'generated_comment': '@' + f'generated from {fm.template_dir_for_comments()}/' + fname, 'autograd_function_declarations': declarations, 'autograd_function_definitions': definitions})