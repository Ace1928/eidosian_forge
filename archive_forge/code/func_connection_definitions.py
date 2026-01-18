from typing import Any, Dict, List, NamedTuple, Optional, Union
from graphql import (
from graphql import GraphQLNamedOutputType
def connection_definitions(node_type: Union[GraphQLNamedOutputType, GraphQLNonNull[GraphQLNamedOutputType]], name: Optional[str]=None, resolve_node: Optional[GraphQLFieldResolver]=None, resolve_cursor: Optional[GraphQLFieldResolver]=None, edge_fields: Optional[ThunkMapping[GraphQLField]]=None, connection_fields: Optional[ThunkMapping[GraphQLField]]=None) -> GraphQLConnectionDefinitions:
    """Return GraphQLObjectTypes for a connection with the given name.

    The nodes of the returned object types will be of the specified type.
    """
    name = name or get_named_type(node_type).name
    edge_type = GraphQLObjectType(name + 'Edge', description='An edge in a connection.', fields=lambda: {'node': GraphQLField(node_type, resolve=resolve_node, description='The item at the end of the edge'), 'cursor': GraphQLField(GraphQLNonNull(GraphQLString), resolve=resolve_cursor, description='A cursor for use in pagination'), **resolve_thunk(edge_fields or {})})
    connection_type = GraphQLObjectType(name + 'Connection', description='A connection to a list of items.', fields=lambda: {'pageInfo': GraphQLField(GraphQLNonNull(page_info_type), description='Information to aid in pagination.'), 'edges': GraphQLField(GraphQLList(edge_type), description='A list of edges.'), **resolve_thunk(connection_fields or {})})
    return GraphQLConnectionDefinitions(edge_type, connection_type)