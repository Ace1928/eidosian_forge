import itertools
from typing import Dict, List, Sequence, Union
from torchgen.api import cpp
from torchgen.api.types import DispatcherSignature
from torchgen.code_template import CodeTemplate
from torchgen.context import with_native_function
from torchgen.model import Argument, NativeFunction, SchemaKind, TensorOptionsArguments
from torchgen.utils import FileManager
def gen_trace_type(out: str, native_functions: List[NativeFunction], template_path: str) -> None:
    fm = FileManager(install_dir=out, template_dir=template_path, dry_run=False)
    fm.write_sharded('TraceType.cpp', [fn for fn in native_functions if cpp.name(fn.func) not in MANUAL_TRACER], key_fn=lambda fn: fn.root_name, base_env={'generated_comment': '@' + f'generated from {fm.template_dir_for_comments()}/TraceType.cpp'}, env_callable=gen_trace_type_func, num_shards=5, sharded_keys={'ops_headers', 'trace_method_definitions', 'trace_wrapper_registrations'})