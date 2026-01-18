from typing import Any, cast
from ....error import GraphQLError
from ....language import ArgumentNode, EnumValueNode, FieldNode, ObjectFieldNode
from ....type import GraphQLInputObjectType, get_named_type, is_input_object_type
from .. import ValidationRule
def enter_argument(self, node: ArgumentNode, *_args: Any) -> None:
    context = self.context
    arg_def = context.get_argument()
    if arg_def:
        deprecation_reason = arg_def.deprecation_reason
        if deprecation_reason is not None:
            directive_def = context.get_directive()
            arg_name = node.name.value
            if directive_def is None:
                parent_type = context.get_parent_type()
                parent_name = parent_type.name
                field_def = context.get_field_def()
                field_name = field_def.ast_node.name.value
                self.report_error(GraphQLError(f"Field '{parent_name}.{field_name}' argument '{arg_name}' is deprecated. {deprecation_reason}", node))
            else:
                self.report_error(GraphQLError(f"Directive '@{directive_def.name}' argument '{arg_name}' is deprecated. {deprecation_reason}", node))