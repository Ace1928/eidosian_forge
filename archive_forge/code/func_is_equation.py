from typing import Optional
from docutils import nodes
from sphinx.builders.html import HTMLTranslator
def is_equation(part: str) -> str:
    return part.strip()