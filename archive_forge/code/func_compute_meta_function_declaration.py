import argparse
import functools
import json
import os
import pathlib
from collections import defaultdict, namedtuple, OrderedDict
from dataclasses import dataclass, field
from typing import (
import yaml
import torchgen.api.dispatcher as dispatcher
import torchgen.api.meta as meta
import torchgen.api.native as native
import torchgen.api.structured as structured
import torchgen.dest as dest
from torchgen.api import cpp
from torchgen.api.translate import translate
from torchgen.api.types import (
from torchgen.context import (
from torchgen.gen_functionalization_type import (
from torchgen.gen_vmap_plumbing import gen_all_vmap_plumbing
from torchgen.model import (
from torchgen.native_function_generation import (
from torchgen.selective_build.selector import SelectiveBuilder
from torchgen.utils import (
from torchgen.yaml_utils import YamlDumper, YamlLoader
def compute_meta_function_declaration(g: NativeFunctionsGroup) -> Optional[str]:
    if not g.structured:
        return None
    with native_function_manager(g.out):
        name = meta.name(g)
        args = structured.meta_arguments(g)
        args_str = ', '.join((a.decl() for a in args))
        parent_class = g.out.structured_inherits
        if parent_class is None:
            parent_class = 'at::impl::MetaBase'
        meta_return = 'void'
        precomputed = g.out.precomputed if g.structured else None
        if precomputed:
            precomputed_values = [*precomputed.replace.values(), precomputed.add]
            precomputed_elements = [elem for replace_list in precomputed_values for elem in replace_list]
            precomputed_template_parameters = [elem.name.upper() for elem in precomputed_elements]
            precomputed_template_params_str = ', '.join((f'bool {param} = false' for param in precomputed_template_parameters))
            precompute_template_decl = f'template <{precomputed_template_params_str}>'
            precomputed_elements_with_cpp_types = [structured.argument_type(elem, binds=elem.name) for elem in precomputed_elements]
            precomputed_elements_decl = ';\n'.join((f'{elem.cpp_type(strip_ref=True)} {elem.name}' for elem in precomputed_elements_with_cpp_types))
            setter_methods = []
            for i, elem in enumerate(precomputed_elements):
                return_ty_templates = ', '.join(precomputed_template_parameters[:i] + ['true'] + precomputed_template_parameters[i + 1:])
                return_ty = f'precompute_out<{return_ty_templates}>'
                elem_cpp_ty = precomputed_elements_with_cpp_types[i].cpp_type(strip_ref=True)
                signature = f'{return_ty} set_{elem.name}({elem_cpp_ty} value)'
                assert_msg = f'"{precomputed_elements[i].name} already set"'
                assert_stmt = f'static_assert({precomputed_template_parameters[i]} == false, {assert_msg});'
                construction_stmts = []
                construction_stmts.append(f'{return_ty} ret;')
                for j, elem in enumerate(precomputed_elements):
                    if i == j:
                        construction_stmts.append(f'ret.{elem.name} = value;')
                    else:
                        construction_stmts.append(f'ret.{elem.name} = this->{elem.name};')
                construction_stmts.append('return ret;')
                construction_block = '\n'.join(construction_stmts)
                setter_methods.append(f'\n                    {signature} {{\n                        {assert_stmt}\n                        {construction_block}\n                    }}\n                ')
            setter_methods_decl = '\n'.join(setter_methods)
            meta_return_template_params = ', '.join(['true'] * len(precomputed_template_parameters))
            meta_return_typedef = f'using meta_return_ty = precompute_out <{meta_return_template_params}>;'
            meta_return = 'meta_return_ty'
            precomputed_decl = f'\n                {precompute_template_decl}\n                struct TORCH_API precompute_out {{\n                    {setter_methods_decl}\n                    {precomputed_elements_decl};\n            }};'
        else:
            meta_return_typedef = ''
            precomputed_decl = ''
        return f'struct TORCH_API structured_{name} : public {parent_class} {{\n    {precomputed_decl}\n    {meta_return_typedef}\n    {meta_return} meta({args_str});\n}};\n'