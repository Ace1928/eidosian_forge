import re
import html
import contextlib
from typing import Optional
def extract_email(text: str) -> Optional[str]:
    """
    Extracts the email
    """
    match = re.search('[\\w\\.-]+@[\\w\\.-]+', text)
    return match[0] if match else None