from io import BytesIO
from . import osutils, progress, trace
from .i18n import gettext
from .ui import ui_factory
def file_match(self, paths):
    """Return a mapping from file_ids to the supplied paths."""
    return self._match_hits(self.get_all_hits(paths))