from .execution import ExecutionResult, execute
from .language.ast import Document
from .language.parser import parse
from .language.source import Source
from .validation import validate
def graphql(schema, request_string='', root_value=None, context_value=None, variable_values=None, operation_name=None, executor=None, return_promise=False, middleware=None):
    try:
        if isinstance(request_string, Document):
            ast = request_string
        else:
            source = Source(request_string, 'GraphQL request')
            ast = parse(source)
        validation_errors = validate(schema, ast)
        if validation_errors:
            return ExecutionResult(errors=validation_errors, invalid=True)
        return execute(schema, ast, root_value, context_value, operation_name=operation_name, variable_values=variable_values or {}, executor=executor, return_promise=return_promise, middleware=middleware)
    except Exception as e:
        return ExecutionResult(errors=[e], invalid=True)