from collections import Counter
import numpy as np
from scipy import sparse
from pygsp import utils
from . import fourier, difference  # prevent circular import in Python < 3.5
def set_coordinates(self, kind='spring', **kwargs):
    """Set node's coordinates (their position when plotting).

        Parameters
        ----------
        kind : string or array-like
            Kind of coordinates to generate. It controls the position of the
            nodes when plotting the graph. Can either pass an array of size Nx2
            or Nx3 to set the coordinates manually or the name of a layout
            algorithm. Available algorithms: community2D, random2D, random3D,
            ring2D, line1D, spring. Default is 'spring'.
        kwargs : dict
            Additional parameters to be passed to the Fruchterman-Reingold
            force-directed algorithm when kind is spring.

        Examples
        --------
        >>> G = graphs.ErdosRenyi()
        >>> G.set_coordinates()
        >>> G.plot()

        """
    if not isinstance(kind, str):
        coords = np.asarray(kind).squeeze()
        check_1d = coords.ndim == 1
        check_2d_3d = coords.ndim == 2 and 2 <= coords.shape[1] <= 3
        if coords.shape[0] != self.N or not (check_1d or check_2d_3d):
            raise ValueError('Expecting coordinates to be of size N, Nx2, or Nx3.')
        self.coords = coords
    elif kind == 'line1D':
        self.coords = np.arange(self.N)
    elif kind == 'line2D':
        x, y = (np.arange(self.N), np.zeros(self.N))
        self.coords = np.stack([x, y], axis=1)
    elif kind == 'ring2D':
        angle = np.arange(self.N) * 2 * np.pi / self.N
        self.coords = np.stack([np.cos(angle), np.sin(angle)], axis=1)
    elif kind == 'random2D':
        self.coords = np.random.uniform(size=(self.N, 2))
    elif kind == 'random3D':
        self.coords = np.random.uniform(size=(self.N, 3))
    elif kind == 'spring':
        self.coords = self._fruchterman_reingold_layout(**kwargs)
    elif kind == 'community2D':
        if not hasattr(self, 'info') or 'node_com' not in self.info:
            ValueError('Missing arguments to the graph to be able to compute community coordinates.')
        if 'world_rad' not in self.info:
            self.info['world_rad'] = np.sqrt(self.N)
        if 'comm_sizes' not in self.info:
            counts = Counter(self.info['node_com'])
            self.info['comm_sizes'] = np.array([cnt[1] for cnt in sorted(counts.items())])
        Nc = self.info['comm_sizes'].shape[0]
        self.info['com_coords'] = self.info['world_rad'] * np.array(list(zip(np.cos(2 * np.pi * np.arange(1, Nc + 1) / Nc), np.sin(2 * np.pi * np.arange(1, Nc + 1) / Nc))))
        coords = np.random.rand(self.N, 2)
        self.coords = np.array([[elem[0] * np.cos(2 * np.pi * elem[1]), elem[0] * np.sin(2 * np.pi * elem[1])] for elem in coords])
        for i in range(self.N):
            comm_idx = self.info['node_com'][i]
            comm_rad = np.sqrt(self.info['comm_sizes'][comm_idx])
            self.coords[i] = self.info['com_coords'][comm_idx] + comm_rad * self.coords[i]
    else:
        raise ValueError('Unexpected argument king={}.'.format(kind))