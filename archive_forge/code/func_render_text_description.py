from typing import Callable, List
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import (
def render_text_description(tools: List[BaseTool]) -> str:
    """Render the tool name and description in plain text.

    Output will be in the format of:

    .. code-block:: markdown

        search: This tool is used for search
        calculator: This tool is used for math
    """
    return '\n'.join([f'{tool.name}: {tool.description}' for tool in tools])