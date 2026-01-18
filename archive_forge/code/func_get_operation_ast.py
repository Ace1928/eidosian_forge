from ..language import ast
def get_operation_ast(document_ast, operation_name=None):
    operation = None
    for definition in document_ast.definitions:
        if isinstance(definition, ast.OperationDefinition):
            if not operation_name:
                if operation:
                    return None
                operation = definition
            elif definition.name and definition.name.value == operation_name:
                return definition
    return operation