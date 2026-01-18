from typing import Any, cast
from ....error import GraphQLError
from ....language import ArgumentNode, EnumValueNode, FieldNode, ObjectFieldNode
from ....type import GraphQLInputObjectType, get_named_type, is_input_object_type
from .. import ValidationRule
def enter_object_field(self, node: ObjectFieldNode, *_args: Any) -> None:
    context = self.context
    input_object_def = get_named_type(context.get_parent_input_type())
    if is_input_object_type(input_object_def):
        input_field_def = cast(GraphQLInputObjectType, input_object_def).fields.get(node.name.value)
        if input_field_def:
            deprecation_reason = input_field_def.deprecation_reason
            if deprecation_reason is not None:
                field_name = node.name.value
                input_object_name = input_object_def.name
                self.report_error(GraphQLError(f'The input field {input_object_name}.{field_name} is deprecated. {deprecation_reason}', node))