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
def on_builder_inited(app: Sphinx) -> None:
    """Initialize DurationDomain on bootstrap.

    This clears results of last build.
    """
    domain = cast(DurationDomain, app.env.get_domain('duration'))
    domain.clear()