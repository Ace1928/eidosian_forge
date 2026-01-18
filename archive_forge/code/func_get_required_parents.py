from io import BytesIO
from . import osutils, progress, trace
from .i18n import gettext
from .ui import ui_factory
def get_required_parents(self, matches):
    """Return a dict of all file parents that must be versioned.

        The keys are the required parents and the values are sets of their
        children.
        """
    required_parents = {}
    for path in matches:
        while True:
            child = path
            path = osutils.dirname(path)
            if self.tree.is_versioned(path):
                break
            required_parents.setdefault(path, []).append(child)
    require_ids = {}
    for parent, children in required_parents.items():
        child_file_ids = set()
        for child in children:
            file_id = matches.get(child)
            if file_id is not None:
                child_file_ids.add(file_id)
        require_ids[parent] = child_file_ids
    return require_ids