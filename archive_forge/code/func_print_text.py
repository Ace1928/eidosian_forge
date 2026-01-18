from typing import Dict, List, Optional, TextIO
def print_text(text: str, color: Optional[str]=None, end: str='', file: Optional[TextIO]=None) -> None:
    """Print text with highlighting and no end characters."""
    text_to_print = get_colored_text(text, color) if color else text
    print(text_to_print, end=end, file=file)
    if file:
        file.flush()