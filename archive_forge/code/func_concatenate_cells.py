import json
from pathlib import Path
from typing import Any, List, Union
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader
def concatenate_cells(cell: dict, include_outputs: bool, max_output_length: int, traceback: bool) -> str:
    """Combine cells information in a readable format ready to be used.

    Args:
        cell: A dictionary
        include_outputs: Whether to include the outputs of the cell.
        max_output_length: Maximum length of the output to be displayed.
        traceback: Whether to return a traceback of the error.

    Returns:
        A string with the cell information.

    """
    cell_type = cell['cell_type']
    source = cell['source']
    if include_outputs:
        try:
            output = cell['outputs']
        except KeyError:
            pass
    if include_outputs and cell_type == 'code' and output:
        if 'ename' in output[0].keys():
            error_name = output[0]['ename']
            error_value = output[0]['evalue']
            if traceback:
                traceback = output[0]['traceback']
                return f"'{cell_type}' cell: '{source}'\n, gives error '{error_name}', with description '{error_value}'\nand traceback '{traceback}'\n\n"
            else:
                return f"'{cell_type}' cell: '{source}'\n, gives error '{error_name}',with description '{error_value}'\n\n"
        elif output[0]['output_type'] == 'stream':
            output = output[0]['text']
            min_output = min(max_output_length, len(output))
            return f"'{cell_type}' cell: '{source}'\n with output: '{output[:min_output]}'\n\n"
    else:
        return f"'{cell_type}' cell: '{source}'\n\n"
    return ''