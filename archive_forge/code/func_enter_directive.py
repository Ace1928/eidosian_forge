from typing import cast, Any, Dict, List, Optional, Tuple, Union
from ...error import GraphQLError
from ...language import (
from ...type import specified_directives
from . import ASTValidationRule, SDLValidationContext, ValidationContext
def enter_directive(self, node: DirectiveNode, _key: Any, _parent: Any, _path: Any, ancestors: List[Node]) -> None:
    name = node.name.value
    locations = self.locations_map.get(name)
    if locations:
        candidate_location = get_directive_location_for_ast_path(ancestors)
        if candidate_location and candidate_location not in locations:
            self.report_error(GraphQLError(f"Directive '@{name}' may not be used on {candidate_location.value}.", node))
    else:
        self.report_error(GraphQLError(f"Unknown directive '@{name}'.", node))