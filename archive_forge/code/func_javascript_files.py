from __future__ import annotations
from pathlib import Path
import tornado.web
def javascript_files(self) -> list[str]:
    """Get the list of JS files to include."""
    return ['/xstatic/termjs/term.js', '/static/terminado.js']