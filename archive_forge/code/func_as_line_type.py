from __future__ import annotations
from contourpy._contourpy import FillType, LineType, ZInterp
def as_line_type(line_type: LineType | str) -> LineType:
    """Coerce a LineType or string value to a LineType.

    Args:
        line_type (LineType or str): Value to convert.

    Return:
        LineType: Converted value.
    """
    if isinstance(line_type, str):
        try:
            return LineType.__members__[line_type]
        except KeyError as e:
            raise ValueError(f"'{line_type}' is not a valid LineType") from e
    else:
        return line_type