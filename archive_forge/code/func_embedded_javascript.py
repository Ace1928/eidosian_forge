from __future__ import annotations
from pathlib import Path
import tornado.web
def embedded_javascript(self) -> str:
    """Get the embedded JS content as a string."""
    file = Path(__file__).parent / 'uimod_embed.js'
    with file.open() as f:
        return f.read()