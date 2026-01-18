import re
import ast
import logging
def extract_function_definitions(self) -> list:
    """
        Uses AST to extract function definitions with detailed logging.

        Returns:
            list: A list of function definitions extracted from the script content using AST.
        """
    self.parser_logger.debug('Attempting to extract function definitions using AST.')
    tree = ast.parse(self.script_content)
    function_definitions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
    self.parser_logger.info(f'Extracted {len(function_definitions)} function definitions.')
    return function_definitions