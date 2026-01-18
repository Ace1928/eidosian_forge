from io import BytesIO
from . import osutils, progress, trace
from .i18n import gettext
from .ui import ui_factory
def match_parents(self, required_parents, missing_parents):
    """Map parent directories to file-ids.

        This is done by finding similarity between the file-ids of children of
        required parent directories and the file-ids of children of missing
        parent directories.
        """
    all_hits = []
    for file_id, file_id_children in missing_parents.items():
        for path, path_children in required_parents.items():
            hits = len(path_children.intersection(file_id_children))
            if hits > 0:
                all_hits.append((hits, path, file_id))
    return self._match_hits(all_hits)