from __future__ import annotations
from lazyops.types import BaseModel, lazyproperty
from fastapi.responses import HTMLResponse
from typing import Optional, Dict, Any
@lazyproperty
def css_url(self):
    """
        Return the CSS URL for the Stoplight Elements.
        """
    return f'https://unpkg.com/@stoplight/elements@{self.version}/styles.min.css' if self.version and self.version != 'latest' else 'https://unpkg.com/@stoplight/elements/styles.min.css'