from ._LinearDrawer import LinearDrawer
from ._CircularDrawer import CircularDrawer
from ._Track import Track
from Bio.Graphics import _write
def renumber_tracks(self, low=1, step=1):
    """Renumber all tracks consecutively.

        Optionally from a passed lowest number.

        Arguments:
         - low     - an integer. The track number to start from.
         - step    - an integer. The track interval for separation of
           tracks.

        """
    track = low
    levels = self.get_levels()
    conversion = {}
    for level in levels:
        conversion[track] = self.tracks[level]
        conversion[track].track_level = track
        track += step
    self.tracks = conversion