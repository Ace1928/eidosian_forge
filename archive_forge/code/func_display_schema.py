import os
from collections import defaultdict
from typing import Any, Dict, List, NamedTuple, Sequence, Set, Tuple
import numpy as np
from onnx import defs, helper
from onnx.backend.sample.ops import collect_sample_implementations
from onnx.backend.test.case import collect_snippets
from onnx.defs import ONNX_ML_DOMAIN, OpSchema
def display_schema(schema: OpSchema, versions: Sequence[OpSchema], changelog: str) -> str:
    s = ''
    if schema.doc:
        s += '\n'
        s += '\n'.join((('  ' + line).rstrip() for line in schema.doc.lstrip().splitlines()))
        s += '\n'
    s += '\n#### Version\n'
    if schema.support_level == OpSchema.SupportType.EXPERIMENTAL:
        s += '\nNo versioning maintained for experimental ops.'
    else:
        s += '\nThis version of the operator has been ' + ('deprecated' if schema.deprecated else 'available') + f' since version {schema.since_version}'
        s += f' of {display_domain(schema.domain)}.\n'
        if len(versions) > 1:
            s += '\nOther versions of this operator: {}\n'.format(', '.join((display_version_link(format_name_with_domain(v.domain, v.name), v.since_version, changelog) for v in versions[:-1])))
    if schema.deprecated:
        return s
    if schema.attributes:
        s += '\n#### Attributes\n\n'
        s += '<dl>\n'
        for _, attr in sorted(schema.attributes.items()):
            opt = ''
            if attr.required:
                opt = 'required'
            elif attr.default_value.name:
                default_value = helper.get_attribute_value(attr.default_value)
                doc_string = attr.default_value.doc_string

                def format_value(value: Any) -> str:
                    if isinstance(value, float):
                        formatted = str(np.round(value, 5))
                        if len(formatted) > 10:
                            formatted = str(f'({value:e})')
                        return formatted
                    if isinstance(value, (bytes, bytearray)):
                        return str(value.decode('utf-8'))
                    return str(value)
                if isinstance(default_value, list):
                    default_value = [format_value(val) for val in default_value]
                else:
                    default_value = format_value(default_value)
                opt = f'default is {default_value}{doc_string}'
            s += f'<dt><tt>{attr.name}</tt> : {display_attr_type(attr.type)}{(f' ({opt})' if opt else '')}</dt>\n'
            s += f'<dd>{attr.description}</dd>\n'
        s += '</dl>\n'
    s += '\n#### Inputs'
    if schema.min_input != schema.max_input:
        s += f' ({display_number(schema.min_input)} - {display_number(schema.max_input)})'
    s += '\n\n'
    if schema.inputs:
        s += '<dl>\n'
        for input_ in schema.inputs:
            option_str = generate_formal_parameter_tags(input_)
            s += f'<dt><tt>{input_.name}</tt>{option_str} : {input_.type_str}</dt>\n'
            s += f'<dd>{input_.description}</dd>\n'
        s += '</dl>\n'
    s += '\n#### Outputs'
    if schema.min_output != schema.max_output:
        s += f' ({display_number(schema.min_output)} - {display_number(schema.max_output)})'
    s += '\n\n'
    if schema.outputs:
        s += '<dl>\n'
        for output in schema.outputs:
            option_str = generate_formal_parameter_tags(output)
            s += f'<dt><tt>{output.name}</tt>{option_str} : {output.type_str}</dt>\n'
            s += f'<dd>{output.description}</dd>\n'
        s += '</dl>\n'
    s += '\n#### Type Constraints'
    s += '\n\n'
    if schema.type_constraints:
        s += '<dl>\n'
        for type_constraint in schema.type_constraints:
            allowedTypes = type_constraint.allowed_type_strs
            if len(allowedTypes) > 0:
                allowedTypeStr = allowedTypes[0]
            for allowedType in allowedTypes[1:]:
                allowedTypeStr += ', ' + allowedType
            s += f'<dt><tt>{type_constraint.type_param_str}</tt> : {allowedTypeStr}</dt>\n'
            s += f'<dd>{type_constraint.description}</dd>\n'
        s += '</dl>\n'
    return s