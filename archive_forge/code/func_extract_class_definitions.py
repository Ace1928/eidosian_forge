import re
import ast
import logging
def extract_class_definitions(self) -> list:
    """
        Uses AST to extract class definitions with detailed logging.

        Returns:
            list: A list of class definitions extracted from the script content using AST.
        """
    self.parser_logger.debug('Attempting to extract class definitions using AST.')
    tree = ast.parse(self.script_content)
    class_definitions = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
    self.parser_logger.info(f'Extracted {len(class_definitions)} class definitions.')
    return class_definitions