from __future__ import annotations
from typing import TYPE_CHECKING, List
from ..iapi import strip_draw_info, strip_label_details
@property
def axs(self) -> list[Axes]:
    return self.facet.axs