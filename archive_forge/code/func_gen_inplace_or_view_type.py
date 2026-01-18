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
def gen_inplace_or_view_type(out: str, native_yaml_path: str, tags_yaml_path: str, fns_with_infos: List[NativeFunctionWithDifferentiabilityInfo], template_path: str) -> None:
    num_shards = 2
    fm = FileManager(install_dir=out, template_dir=template_path, dry_run=False)
    fm.write_sharded('ADInplaceOrViewType.cpp', [fn for fn in fns_with_infos if use_derived(fn)], key_fn=lambda fn: fn.func.root_name, base_env={'generated_comment': '@' + f'generated from {fm.template_dir_for_comments()}/ADInplaceOrViewType.cpp'}, env_callable=gen_inplace_or_view_type_env, num_shards=2, sharded_keys={'ops_headers', 'inplace_or_view_method_definitions', 'inplace_or_view_wrapper_registrations'})