from typing import Callable, Dict, List, Optional, Union, Tuple
from graphql import GraphQLError
from graphql.validation import ValidationContext, ValidationRule
from graphql.language import (
from ..utils.is_introspection_key import is_introspection_key
def determine_depth(node: Node, fragments: Dict[str, FragmentDefinitionNode], depth_so_far: int, max_depth: int, context: ValidationContext, operation_name: str, ignore: Optional[List[IgnoreType]]=None) -> int:
    if depth_so_far > max_depth:
        context.report_error(GraphQLError(f"'{operation_name}' exceeds maximum operation depth of {max_depth}.", [node]))
        return depth_so_far
    if isinstance(node, FieldNode):
        should_ignore = is_introspection_key(node.name.value) or is_ignored(node, ignore)
        if should_ignore or not node.selection_set:
            return 0
        return 1 + max(map(lambda selection: determine_depth(node=selection, fragments=fragments, depth_so_far=depth_so_far + 1, max_depth=max_depth, context=context, operation_name=operation_name, ignore=ignore), node.selection_set.selections))
    elif isinstance(node, FragmentSpreadNode):
        return determine_depth(node=fragments[node.name.value], fragments=fragments, depth_so_far=depth_so_far, max_depth=max_depth, context=context, operation_name=operation_name, ignore=ignore)
    elif isinstance(node, (InlineFragmentNode, FragmentDefinitionNode, OperationDefinitionNode)):
        return max(map(lambda selection: determine_depth(node=selection, fragments=fragments, depth_so_far=depth_so_far, max_depth=max_depth, context=context, operation_name=operation_name, ignore=ignore), node.selection_set.selections))
    else:
        raise Exception(f'Depth crawler cannot handle: {node.kind}.')