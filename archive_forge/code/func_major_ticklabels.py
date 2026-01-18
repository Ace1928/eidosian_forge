import matplotlib.axes as maxes
from matplotlib.artist import Artist
from matplotlib.axis import XAxis, YAxis
@property
def major_ticklabels(self):
    label = 'label%d' % self._axisnum
    return SimpleChainedObjects([getattr(tick, label) for tick in self._axis.get_major_ticks()])