import re
from typing import List, Optional
import torchgen.api.python as python
from torchgen.api import cpp
from torchgen.api.types import CppSignatureGroup
from torchgen.context import with_native_function
from torchgen.gen import parse_native_yaml
from torchgen.model import NativeFunction, TensorOptionsArguments, Variant
from torchgen.utils import FileManager, mapMaybe
def gen_variable_factories(out: str, native_yaml_path: str, tags_yaml_path: str, template_path: str) -> None:
    native_functions = parse_native_yaml(native_yaml_path, tags_yaml_path).native_functions
    factory_functions = [fn for fn in native_functions if is_factory_function(fn)]
    fm = FileManager(install_dir=out, template_dir=template_path, dry_run=False)
    fm.write_with_template('variable_factories.h', 'variable_factories.h', lambda: {'generated_comment': '@' + f'generated from {fm.template_dir_for_comments()}/variable_factories.h', 'ops_headers': [f'#include <ATen/ops/{fn.root_name}.h>' for fn in factory_functions], 'function_definitions': list(mapMaybe(process_function, factory_functions))})