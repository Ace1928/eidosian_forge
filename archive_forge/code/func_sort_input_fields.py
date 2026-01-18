from typing import Collection, Dict, Optional, Tuple, Union, cast
from ..language import DirectiveLocation
from ..pyutils import inspect, merge_kwargs, natural_comparison_key
from ..type import (
def sort_input_fields(fields_map: Dict[str, GraphQLInputField]) -> Dict[str, GraphQLInputField]:
    return {name: GraphQLInputField(cast(GraphQLInputType, replace_type(cast(GraphQLNamedType, field.type))), description=field.description, default_value=field.default_value, ast_node=field.ast_node) for name, field in sorted(fields_map.items())}