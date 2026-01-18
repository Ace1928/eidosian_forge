import itertools
import re
from collections import defaultdict
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple
import yaml
from torchgen.api import cpp
from torchgen.api.python import (
from torchgen.code_template import CodeTemplate
from torchgen.context import with_native_function
from torchgen.gen import cpp_string, parse_native_yaml, parse_tags_yaml
from torchgen.model import (
from torchgen.utils import FileManager, split_name_params
from torchgen.yaml_utils import YamlLoader
from .gen_trace_type import should_trace
def is_schema_compatible(aten_schema: FunctionSchema) -> bool:
    arguments: Iterable[Argument]
    if is_out:
        arguments = itertools.chain(aten_schema.arguments.out, aten_schema.arguments.flat_non_out)
    else:
        arguments = aten_schema.arguments.flat_all
    for i, arg in enumerate(arguments):
        if i < len(call_args):
            arg_name = call_args[i]
            if arg_name in known_constants:
                schema_type = known_constants[arg_name]
                schema_annotation = None
            else:
                schema_arg = schema_args_by_name[arg_name]
                schema_type = schema_arg.type
                schema_annotation = schema_arg.annotation
            if schema_type != arg.type or schema_annotation != arg.annotation:
                return False
        elif arg.default is None:
            return False
    return len(schema.returns) == len(aten_schema.returns) and all((a == b for a, b in zip(schema.returns, aten_schema.returns)))