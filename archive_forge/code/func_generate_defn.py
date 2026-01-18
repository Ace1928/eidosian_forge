import itertools
import textwrap
from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple, Union
import torchgen.api.cpp as cpp
import torchgen.api.meta as meta
import torchgen.api.structured as structured
from torchgen.api.translate import translate
from torchgen.api.types import (
from torchgen.context import method_with_native_function, native_function_manager
from torchgen.model import (
from torchgen.selective_build.selector import SelectiveBuilder
from torchgen.utils import assert_never, mapMaybe, Target
def generate_defn(cpp_sig: CppSignature) -> str:
    return f'\n{cpp_sig.defn()} {{\nreturn {sig.name()}({', '.join((e.expr for e in translate(cpp_sig.arguments(), sig.arguments())))});\n}}\n'