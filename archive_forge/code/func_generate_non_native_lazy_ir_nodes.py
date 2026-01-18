import itertools
from abc import ABC
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import torchgen.api.dispatcher as dispatcher
from torchgen.api.lazy import (
from torchgen.api.translate import translate
from torchgen.api.types import (
from torchgen.context import method_with_native_function
from torchgen.dest.lazy_ts_lowering import ts_lowering_body
from torchgen.model import (
def generate_non_native_lazy_ir_nodes(non_native: List[Dict[str, Any]], gen_lazy_ir: GenLazyIR) -> List[str]:
    """Generate the non-native lazy IR node classes"""
    nodes = []
    for op in non_native:
        properties = LazyIrProperties('ShapeCache', 'CanBeReused', 'LowerDeclOnly')
        for p in op.get('properties', []):
            setattr(properties, p, True)
        schema = LazyIrSchema(FunctionSchema.parse(op['func']), properties, symint=True)
        schema.opkind = op.get('opkind')
        nodes.append(gen_lazy_ir.gen(schema)[0])
    return nodes