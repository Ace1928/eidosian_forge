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
def note_reading_duration(self, duration: timedelta) -> None:
    self.reading_durations[self.env.docname] = duration