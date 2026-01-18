import collections
import itertools
from numbers import Number
import networkx as nx
from networkx.drawing.layout import (
def self_loop_connection(posA, posB, *args, **kwargs):
    if not self.np.all(posA == posB):
        raise nx.NetworkXError('`self_loop` connection style methodis only to be used for self-loops')
    data_loc = self.ax.transData.inverted().transform(posA)
    v_shift = 0.1 * self.selfloop_height
    h_shift = v_shift * 0.5
    path = self.np.asarray([[0, v_shift], [h_shift, v_shift], [h_shift, 0], [0, 0], [-h_shift, 0], [-h_shift, v_shift], [0, v_shift]])
    if edge_index % 4:
        x, y = path.T
        for _ in range(edge_index % 4):
            x, y = (y, -x)
        path = self.np.array([x, y]).T
    return self.mpl.path.Path(self.ax.transData.transform(data_loc + path), [1, 4, 4, 4, 4, 4, 4])