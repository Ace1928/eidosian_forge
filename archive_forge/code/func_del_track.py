from ._LinearDrawer import LinearDrawer
from ._CircularDrawer import CircularDrawer
from ._Track import Track
from Bio.Graphics import _write
def del_track(self, track_level):
    """Remove the track to be drawn at a particular level on the diagram.

        Arguments:
            - track_level   - an integer. The level of the track on the diagram
              to delete.

        del_track(self, track_level)
        """
    del self.tracks[track_level]