from ...error import GraphQLError
from ...type.definition import GraphQLNonNull
from ...utils.type_comparators import is_type_sub_type_of
from ...utils.type_from_ast import type_from_ast
from .base import ValidationRule
def leave_OperationDefinition(self, operation, key, parent, path, ancestors):
    usages = self.context.get_recursive_variable_usages(operation)
    for usage in usages:
        node = usage.node
        type = usage.type
        var_name = node.name.value
        var_def = self.var_def_map.get(var_name)
        if var_def and type:
            schema = self.context.get_schema()
            var_type = type_from_ast(schema, var_def.type)
            if var_type and (not is_type_sub_type_of(schema, self.effective_type(var_type, var_def), type)):
                self.context.report_error(GraphQLError(self.bad_var_pos_message(var_name, var_type, type), [var_def, node]))