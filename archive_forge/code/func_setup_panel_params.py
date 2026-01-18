from __future__ import annotations
import typing
from types import SimpleNamespace
from ..iapi import panel_view
from ..positions.position import transform_position
from .coord import coord, dist_euclidean
def setup_panel_params(self, scale_x: scale, scale_y: scale) -> panel_view:
    """
        Compute the range and break information for the panel
        """
    from mizani.transforms import identity_trans

    def get_scale_view(scale: scale, coord_limits: TupleFloat2) -> scale_view:
        expansion = scale.default_expansion(expand=self.expand)
        ranges = scale.expand_limits(scale.limits, expansion, coord_limits, identity_trans)
        sv = scale.view(limits=coord_limits, range=ranges.range)
        return sv
    out = panel_view(x=get_scale_view(scale_x, self.limits.x), y=get_scale_view(scale_y, self.limits.y))
    return out