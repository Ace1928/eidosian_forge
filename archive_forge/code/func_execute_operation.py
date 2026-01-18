from wandb_promise import Promise
from ...type import GraphQLSchema
from ..base import ExecutionContext, ExecutionResult, get_operation_root_type
from ..executors.sync import SyncExecutor
from ..middleware import MiddlewareManager
from .fragment import Fragment
def execute_operation(exe_context, operation, root_value):
    type = get_operation_root_type(exe_context.schema, operation)
    execute_serially = operation.operation == 'mutation'
    fragment = Fragment(type=type, field_asts=[operation], context=exe_context)
    if execute_serially:
        return fragment.resolve_serially(root_value)
    return fragment.resolve(root_value)