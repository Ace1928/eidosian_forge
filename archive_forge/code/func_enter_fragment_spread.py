from typing import Any
from ...error import GraphQLError
from ...language import FragmentSpreadNode
from . import ValidationRule
def enter_fragment_spread(self, node: FragmentSpreadNode, *_args: Any) -> None:
    fragment_name = node.name.value
    fragment = self.context.get_fragment(fragment_name)
    if not fragment:
        self.report_error(GraphQLError(f"Unknown fragment '{fragment_name}'.", node.name))