from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union
import torchgen.api.ufunc as ufunc
from torchgen.api.translate import translate
from torchgen.api.types import (
from torchgen.api.ufunc import UfunctorBindings
from torchgen.context import with_native_function
from torchgen.model import (
from torchgen.utils import OrderedSet
@with_native_function
def compute_ufunc_cpu(g: NativeFunctionsGroup) -> str:
    stub_sig = StubSignature(g)
    sig = StructuredImplSignature(g, ufunc.kernel_name(g, DispatchKey.CPU))
    return f'\n{stub_sig.type_defn()};\n{stub_sig.dispatch_decl()};\n{stub_sig.dispatch_defn()};\n\n{sig.defn()} {{\n  {stub_sig.call(sig.arguments())};\n}}\n'