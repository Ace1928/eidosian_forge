from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import logging
from pathlib import Path
def join_enhanced_text(enhanced_chars: EnhancedText) -> str:
    """
    Joins the enhanced ASCII art representations of characters into a single formatted string.

    Args:
        enhanced_chars (EnhancedText): A list of enhanced ASCII art representations for each character.

    Returns:
        str: The formatted string containing the enhanced ASCII art text.
    """
    formatted_text: List[str] = []
    for i in range(3):
        row: List[str] = [char[i] for char in enhanced_chars]
        formatted_text.append(''.join(row))
    return '\n'.join(formatted_text)