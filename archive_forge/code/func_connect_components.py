from collections import defaultdict
import networkx as nx
def connect_components(self, v, w):
    """Adds half-edges for (v, w) and (w, v) at some position.

        This method should only be called if v and w are in different
        components, or it might break the embedding.
        This especially means that if `connect_components(v, w)`
        is called it is not allowed to call `connect_components(w, v)`
        afterwards. The neighbor orientations in both directions are
        all set correctly after the first call.

        Parameters
        ----------
        v : node
        w : node

        See Also
        --------
        add_half_edge_ccw
        add_half_edge_cw
        add_half_edge_first
        """
    self.add_half_edge_first(v, w)
    self.add_half_edge_first(w, v)