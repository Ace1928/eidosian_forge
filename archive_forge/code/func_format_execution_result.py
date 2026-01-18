from graphql.error import GraphQLError
from graphene.types.schema import Schema
def format_execution_result(execution_result, format_error):
    if execution_result:
        response = {}
        if execution_result.errors:
            response['errors'] = [format_error(e) for e in execution_result.errors]
        response['data'] = execution_result.data
        return response