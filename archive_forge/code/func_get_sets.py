from reportlab.lib import colors
from ._FeatureSet import FeatureSet
from ._GraphSet import GraphSet
def get_sets(self):
    """Return the sets contained in this track."""
    return list(self._sets.values())