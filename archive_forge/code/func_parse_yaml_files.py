import argparse
import os
import pathlib
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, TextIO, Tuple, Union
import yaml
from torchgen import dest
from torchgen.api import cpp as aten_cpp
from torchgen.api.types import CppSignature, CppSignatureGroup, CType, NamedCType
from torchgen.context import (
from torchgen.executorch.api import et_cpp
from torchgen.executorch.api.custom_ops import (
from torchgen.executorch.api.types import contextArg, ExecutorchCppSignature
from torchgen.executorch.api.unboxing import Unboxing
from torchgen.executorch.model import ETKernelIndex, ETKernelKey, ETParsedYaml
from torchgen.executorch.parse import ET_FIELDS, parse_et_yaml, parse_et_yaml_struct
from torchgen.gen import (
from torchgen.model import (
from torchgen.selective_build.selector import SelectiveBuilder
from torchgen.utils import (
def parse_yaml_files(tags_yaml_path: str, aten_yaml_path: str, native_yaml_path: Optional[str], custom_ops_yaml_path: Optional[str], selector: SelectiveBuilder, use_aten_lib: bool) -> Tuple[ETParsedYaml, Optional[ETParsedYaml]]:
    """Parses functions.yaml and custom_ops.yaml files.

    Args:
        tags_yaml_path: Path to a tags.yaml file to satisfy codegen parsing.
            It is not optional.
        aten_yaml_path: Path to ATen operator yaml file native_functions.yaml.
        native_yaml_path: Path to a functions.yaml file to parse.
            If the path does not exist in the filesystem, it is treated as an
            empty file. If `custom_ops_yaml_path` exists, the contents of that
            file are appended to the yaml input to be parsed.
        custom_ops_yaml_path: Path to a custom_ops.yaml file to parse. If
            the path does not exist in the filesystem, it is ignored.
        selector: For selective build.
        use_aten_lib: We use this flag to determine if we want to generate native
            functions. In ATen mode we should generate out= variants.
    Returns:
        A tuple with two elements:
        [0]: The parsed results of concatenating the contents of
             `native_yaml_path` and `custom_ops_yaml_path`.
        [1]: The parsed results of the contents of `custom_ops_yaml_path`, if
             present. If not present, None.
    """
    import tempfile

    def function_filter(f: NativeFunction) -> bool:
        return selector.is_native_function_selected(f)
    with tempfile.TemporaryDirectory() as tmpdirname:
        translated_yaml_path = os.path.join(tmpdirname, 'translated.yaml')
        with open(translated_yaml_path, 'w') as translated:
            translate_native_yaml(tags_yaml_path, aten_yaml_path, native_yaml_path, use_aten_lib, translated)
        translated_functions, translated_indices = parse_yaml(translated_yaml_path, tags_yaml_path, function_filter, not use_aten_lib)
        custom_ops_functions, custom_ops_indices = parse_yaml(custom_ops_yaml_path, tags_yaml_path, function_filter, True)
        if not isinstance(translated_indices, ETKernelIndex):
            translated_indices = ETKernelIndex.from_backend_indices(translated_indices)
        if not isinstance(custom_ops_indices, ETKernelIndex):
            custom_ops_indices = ETKernelIndex.from_backend_indices(custom_ops_indices)
        combined_functions = translated_functions + custom_ops_functions
        combined_kernel_index = ETKernelIndex.merge_indices(translated_indices, custom_ops_indices)
        combined_yaml = ETParsedYaml(combined_functions, combined_kernel_index)
        custom_ops_parsed_yaml = ETParsedYaml(custom_ops_functions, custom_ops_indices)
    return (combined_yaml, custom_ops_parsed_yaml)