from datetime import datetime, timedelta
from itertools import islice
from operator import itemgetter
from typing import Any, Dict, List, cast
from docutils import nodes
import sphinx
from sphinx.application import Sphinx
from sphinx.domains import Domain
from sphinx.locale import __
from sphinx.util import logging
def on_source_read(app: Sphinx, docname: str, content: List[str]) -> None:
    """Start to measure reading duration."""
    app.env.temp_data['started_at'] = datetime.now()