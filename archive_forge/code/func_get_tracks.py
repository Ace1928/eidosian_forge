from ._LinearDrawer import LinearDrawer
from ._CircularDrawer import CircularDrawer
from ._Track import Track
from Bio.Graphics import _write
def get_tracks(self):
    """Return a list of the tracks contained in the diagram."""
    return list(self.tracks.values())