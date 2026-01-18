from typing import Collection, Dict, Optional, Tuple, Union, cast
from ..language import DirectiveLocation
from ..pyutils import inspect, merge_kwargs, natural_comparison_key
from ..type import (
def sort_by_name_key(type_: Union[GraphQLNamedType, GraphQLDirective, DirectiveLocation]) -> Tuple:
    return natural_comparison_key(type_.name)