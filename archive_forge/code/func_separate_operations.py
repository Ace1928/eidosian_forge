from typing import Any, Dict, List, Set
from ..language import (
def separate_operations(document_ast: DocumentNode) -> Dict[str, DocumentNode]:
    """Separate operations in a given AST document.

    This function accepts a single AST document which may contain many operations and
    fragments and returns a collection of AST documents each of which contains a single
    operation as well the fragment definitions it refers to.
    """
    operations: List[OperationDefinitionNode] = []
    dep_graph: DepGraph = {}
    for definition_node in document_ast.definitions:
        if isinstance(definition_node, OperationDefinitionNode):
            operations.append(definition_node)
        elif isinstance(definition_node, FragmentDefinitionNode):
            dep_graph[definition_node.name.value] = collect_dependencies(definition_node.selection_set)
    separated_document_asts: Dict[str, DocumentNode] = {}
    for operation in operations:
        dependencies: Set[str] = set()
        for fragment_name in collect_dependencies(operation.selection_set):
            collect_transitive_dependencies(dependencies, dep_graph, fragment_name)
        operation_name = operation.name.value if operation.name else ''
        separated_document_asts[operation_name] = DocumentNode(definitions=[node for node in document_ast.definitions if node is operation or (isinstance(node, FragmentDefinitionNode) and node.name.value in dependencies)])
    return separated_document_asts