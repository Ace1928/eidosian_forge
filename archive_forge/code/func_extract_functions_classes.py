import ast
from typing import Any, List, Tuple
from langchain_community.document_loaders.parsers.language.code_segmenter import (
def extract_functions_classes(self) -> List[str]:
    tree = ast.parse(self.code)
    functions_classes = []
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            functions_classes.append(self._extract_code(node))
    return functions_classes