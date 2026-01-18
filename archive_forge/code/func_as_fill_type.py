from __future__ import annotations
from contourpy._contourpy import FillType, LineType, ZInterp
def as_fill_type(fill_type: FillType | str) -> FillType:
    """Coerce a FillType or string value to a FillType.

    Args:
        fill_type (FillType or str): Value to convert.

    Return:
        FillType: Converted value.
    """
    if isinstance(fill_type, str):
        try:
            return FillType.__members__[fill_type]
        except KeyError as e:
            raise ValueError(f"'{fill_type}' is not a valid FillType") from e
    else:
        return fill_type