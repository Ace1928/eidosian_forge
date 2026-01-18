import re
from collections import defaultdict
from typing import Any, Counter, Dict, List, Match, Optional, Sequence, Set, Tuple
import yaml
from torchgen.api import cpp
from torchgen.api.autograd import (
from torchgen.api.types import (
from torchgen.context import with_native_function
from torchgen.gen import get_grouped_by_view_native_functions, parse_native_yaml
from torchgen.model import (
from torchgen.utils import concatMap, IDENT_REGEX, split_name_params
from torchgen.yaml_utils import YamlLoader
def saved_variables(formula: str, nctypes: List[NamedCType], var_names: Tuple[str, ...]) -> Tuple[str, Tuple[SavedAttribute, ...]]:

    def stride_expr(name: str) -> str:
        assert var_names == (name,), 'Replacement for ".strides()" is currently only supported for single derivatives of the same tensor that ".strides()" is being called on.'
        return f'strides_or_error({name}, "{name}")'
    REPLACEMENTS: List[Tuple[str, Dict[str, Any]]] = [('{}.sym_sizes\\(\\)', {'suffix': '_sym_sizes', 'nctype': lambda name: NamedCType(name, BaseCType(symIntArrayRefT))}), ('{}->sym_sizes\\(\\)', {'suffix': '_sym_sizes_opt', 'nctype': lambda name: NamedCType(name, OptionalCType(BaseCType(symIntArrayRefT))), 'expr': lambda name: f'{name}.has_value() ? c10::optional<c10::SymIntArrayRef>({name}->sym_sizes()) : c10::nullopt'}), ('{}.sym_blocksize\\(\\)', {'suffix': '_self_sym_blocksize_opt', 'nctype': lambda name: NamedCType(name, OptionalCType(BaseCType(symIntArrayRefT))), 'expr': lambda name: f'at::sparse_csr::getSymIntBlockSize({name})'}), ('{}.options\\(\\)', {'suffix': '_options', 'nctype': lambda name: NamedCType(name, BaseCType(tensorOptionsT))}), ('zeros_like\\({}\\)', {'suffix': '_info', 'nctype': lambda name: NamedCType(name, BaseCType(typeAndSizeT)), 'expr': lambda name: name, 'res': lambda name: name + '_info.zeros()'}), ('{}.sym_size\\((-?\\w+)\\)', {'suffix': lambda m: f'_sym_argsize_{m.groups()[0].replace('-', 'minus_')}', 'nctype': lambda name: NamedCType(name, BaseCType(SymIntT))}), ('{}.numel\\(\\)', {'suffix': '_numel', 'nctype': lambda name: NamedCType(name, BaseCType(longT))}), ('{}.sym_numel\\(\\)', {'suffix': '_sym_numel', 'nctype': lambda name: NamedCType(name, BaseCType(SymIntT))}), ('to_args_sizes\\({}\\)', {'suffix': '_args_sizes', 'nctype': lambda name: NamedCType(name, VectorCType(VectorCType(BaseCType(longT))))}), ('to_args_sizes_symint\\({}\\)', {'suffix': '_args_sizes_symint', 'nctype': lambda name: NamedCType(name, VectorCType(VectorCType(BaseCType(SymIntT))))}), ('to_args_scalartypes\\({}\\)', {'suffix': '_args_scalartypes', 'nctype': lambda name: NamedCType(name, VectorCType(BaseCType(scalarTypeT)))}), ('TensorGeometry\\({}\\)', {'suffix': '_geometry', 'nctype': lambda name: NamedCType(name, BaseCType(tensorGeometryT))}), ('{}.scalar_type\\(\\)', {'suffix': '_scalar_type', 'nctype': lambda name: NamedCType(name, BaseCType(scalarTypeT))}), ('{}.dim\\(\\)', {'suffix': '_dim', 'nctype': lambda name: NamedCType(name, BaseCType(longT))}), ('{}.sym_strides\\(\\)', {'suffix': '_sym_strides', 'nctype': lambda name: NamedCType(name, BaseCType(symIntArrayRefT)), 'expr': stride_expr}), ('{}.layout\\(\\)', {'suffix': '_layout', 'nctype': lambda name: NamedCType(name, BaseCType(layoutT))}), ('{}.is_conj\\(\\)', {'suffix': '_conjugate', 'nctype': lambda name: NamedCType(name, BaseCType(boolT))})]
    saved: List[SavedAttribute] = []
    if '.sizes()' in formula or '->sizes()' in formula:
        raise RuntimeError('.sizes() is not supported in derivative formulas. Instead, please use the SymInt version,' + f'.sym_sizes(), which returned a c10::SymIntArrayRef. formula={formula}')
    if re.search('\\.size\\([-]?\\d+\\)', formula) or re.search('->size\\([-]?\\d+\\)', formula):
        raise RuntimeError('.size(int) is not supported in derivative formulas. Instead, please use the SymInt version,' + f'.sym_size(int), which returned a c10::SymIntArrayRef. formula={formula}')
    if '.strides()' in formula or '->strides()' in formula:
        raise RuntimeError('.strides() is not supported in derivative formulas. Instead, please use the SymInt version,' + f'.sym_strides(), which returned a c10::SymIntArrayRef. formula={formula}')
    for nctype in nctypes:
        name = nctype.name.name if isinstance(nctype.name, SpecialArgName) else nctype.name
        for regex, info in REPLACEMENTS:

            def repl(m: Match[str]) -> str:
                suffix: str = info['suffix'](m) if callable(info['suffix']) else info['suffix']
                expr: str = info['expr'](name) if 'expr' in info else m.group(0)
                saved.append(SavedAttribute(nctype=info['nctype'](name + suffix), expr=expr))
                if 'res' in info:
                    replacement: str = info['res'](name)
                    return replacement
                return name + suffix
            formula = re.sub(regex.format(name), repl, formula)
        if nctype.type == OptionalCType(BaseCType(stringT)):
            formula = re.sub(f'\\b{name}\\b', f'{name}.has_value() ? c10::optional<c10::string_view>({name}.value()) : c10::nullopt', formula)
        if re.search(IDENT_REGEX.format(name), formula):
            saved.append(SavedAttribute(nctype=nctype, expr=name))
    return (formula, tuple(saved))