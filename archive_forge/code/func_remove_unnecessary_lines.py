from typing import Any, Iterator, List, Sequence, cast
from langchain_core.documents import BaseDocumentTransformer, Document
@staticmethod
def remove_unnecessary_lines(content: str) -> str:
    """
        Clean up the content by removing unnecessary lines.

        Args:
            content: A string, which may contain unnecessary lines or spaces.

        Returns:
            A cleaned string with unnecessary lines removed.
        """
    lines = content.split('\n')
    stripped_lines = [line.strip() for line in lines]
    non_empty_lines = [line for line in stripped_lines if line]
    cleaned_content = ' '.join(non_empty_lines)
    return cleaned_content