from __future__ import annotations
from lazyops.types import BaseModel, lazyproperty
from fastapi.responses import HTMLResponse
from typing import Optional, Dict, Any
@lazyproperty
def js_url(self):
    """
        Return the JS URL for the Stoplight Elements.
        """
    return f'https://unpkg.com/@stoplight/elements@{self.version}/web-components.min.js' if self.version and self.version != 'latest' else 'https://unpkg.com/@stoplight/elements/web-components.min.js'