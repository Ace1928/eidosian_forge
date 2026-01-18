from collections.abc import MutableMapping
import functools
import numpy as np
import matplotlib as mpl
from matplotlib import _api, _docstring
from matplotlib.artist import allow_rasterization
import matplotlib.transforms as mtransforms
import matplotlib.patches as mpatches
import matplotlib.path as mpath
def get_spine_transform(self):
    """Return the spine transform."""
    self._ensure_position_is_set()
    position = self._position
    if isinstance(position, str):
        if position == 'center':
            position = ('axes', 0.5)
        elif position == 'zero':
            position = ('data', 0)
    assert len(position) == 2, 'position should be 2-tuple'
    position_type, amount = position
    _api.check_in_list(['axes', 'outward', 'data'], position_type=position_type)
    if self.spine_type in ['left', 'right']:
        base_transform = self.axes.get_yaxis_transform(which='grid')
    elif self.spine_type in ['top', 'bottom']:
        base_transform = self.axes.get_xaxis_transform(which='grid')
    else:
        raise ValueError(f'unknown spine spine_type: {self.spine_type!r}')
    if position_type == 'outward':
        if amount == 0:
            return base_transform
        else:
            offset_vec = {'left': (-1, 0), 'right': (1, 0), 'bottom': (0, -1), 'top': (0, 1)}[self.spine_type]
            offset_dots = amount * np.array(offset_vec) / 72
            return base_transform + mtransforms.ScaledTranslation(*offset_dots, self.figure.dpi_scale_trans)
    elif position_type == 'axes':
        if self.spine_type in ['left', 'right']:
            return mtransforms.Affine2D.from_values(0, 0, 0, 1, amount, 0) + base_transform
        elif self.spine_type in ['bottom', 'top']:
            return mtransforms.Affine2D.from_values(1, 0, 0, 0, 0, amount) + base_transform
    elif position_type == 'data':
        if self.spine_type in ('right', 'top'):
            amount -= 1
        if self.spine_type in ('left', 'right'):
            return mtransforms.blended_transform_factory(mtransforms.Affine2D().translate(amount, 0) + self.axes.transData, self.axes.transData)
        elif self.spine_type in ('bottom', 'top'):
            return mtransforms.blended_transform_factory(self.axes.transData, mtransforms.Affine2D().translate(0, amount) + self.axes.transData)