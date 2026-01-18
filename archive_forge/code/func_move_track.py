from ._LinearDrawer import LinearDrawer
from ._CircularDrawer import CircularDrawer
from ._Track import Track
from Bio.Graphics import _write
def move_track(self, from_level, to_level):
    """Move a track from one level on the diagram to another.

        Arguments:
         - from_level   - an integer. The level at which the track to be
           moved is found.
         - to_level     - an integer. The level to move the track to.

        """
    aux = self.tracks[from_level]
    del self.tracks[from_level]
    self.add_track(aux, to_level)