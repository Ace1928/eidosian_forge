from io import StringIO
from breezy import osutils, trace
from .bzr.inventorytree import InventoryTreeChange
def get_changes_as_text(self, show_ids=False, show_unchanged=False, short_status=False):
    output = StringIO()
    report_delta(output, self, short_status, show_ids, show_unchanged)
    return output.getvalue()