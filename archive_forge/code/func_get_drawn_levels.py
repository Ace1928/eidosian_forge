from ._LinearDrawer import LinearDrawer
from ._CircularDrawer import CircularDrawer
from ._Track import Track
from Bio.Graphics import _write
def get_drawn_levels(self):
    """Return a sorted list of levels occupied by tracks.

        These tracks are not explicitly hidden.
        """
    return sorted((key for key in self.tracks if not self.tracks[key].hide))