from .. import ui
from ..branch import Branch
from ..check import Check
from ..i18n import gettext
from ..revision import NULL_REVISION
from ..trace import note
from ..workingtree import WorkingTree
def check_weaves(self):
    """Check all the weaves we can get our hands on.
        """
    weave_ids = []
    with ui.ui_factory.nested_progress_bar() as storebar:
        self._check_weaves(storebar)