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
def on_build_finished(app: Sphinx, error: Exception) -> None:
    """Display duration ranking on current build."""
    domain = cast(DurationDomain, app.env.get_domain('duration'))
    durations = sorted(domain.reading_durations.items(), key=itemgetter(1), reverse=True)
    if not durations:
        return
    logger.info('')
    logger.info(__('====================== slowest reading durations ======================='))
    for docname, d in islice(durations, 5):
        logger.info('%d.%03d %s', d.seconds, d.microseconds / 1000, docname)