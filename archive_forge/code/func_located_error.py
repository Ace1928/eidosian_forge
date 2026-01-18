from typing import TYPE_CHECKING, Collection, Optional, Union
from ..pyutils import inspect
from .graphql_error import GraphQLError
def located_error(original_error: Exception, nodes: Optional[Union['None', Collection['Node']]]=None, path: Optional[Collection[Union[str, int]]]=None) -> GraphQLError:
    """Located GraphQL Error

    Given an arbitrary Exception, presumably thrown while attempting to execute a
    GraphQL operation, produce a new GraphQLError aware of the location in the document
    responsible for the original Exception.
    """
    if not isinstance(original_error, Exception):
        original_error = TypeError(f'Unexpected error value: {inspect(original_error)}')
    if isinstance(original_error, GraphQLError) and original_error.path is not None:
        return original_error
    try:
        message = str(original_error.message)
    except AttributeError:
        message = str(original_error)
    try:
        source = original_error.source
    except AttributeError:
        source = None
    try:
        positions = original_error.positions
    except AttributeError:
        positions = None
    try:
        nodes = original_error.nodes or nodes
    except AttributeError:
        pass
    return GraphQLError(message, nodes, source, positions, path, original_error)