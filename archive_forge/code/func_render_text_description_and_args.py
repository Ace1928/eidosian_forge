from typing import Callable, List
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import (
def render_text_description_and_args(tools: List[BaseTool]) -> str:
    """Render the tool name, description, and args in plain text.

    Output will be in the format of:

    .. code-block:: markdown

        search: This tool is used for search, args: {"query": {"type": "string"}}
        calculator: This tool is used for math, args: {"expression": {"type": "string"}}
    """
    tool_strings = []
    for tool in tools:
        args_schema = str(tool.args)
        tool_strings.append(f'{tool.name}: {tool.description}, args: {args_schema}')
    return '\n'.join(tool_strings)