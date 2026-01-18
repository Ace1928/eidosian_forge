from typing import List, NamedTuple, Optional
def box_draw_character(first: Optional[BoxDrawCharacterSet], second: BoxDrawCharacterSet, *, top: int=0, bottom: int=0, left: int=0, right: int=0) -> Optional[str]:
    """Finds a box drawing character based on its connectivity.

    For example:

        box_draw_character(
            NORMAL_BOX_CHARS,
            BOLD_BOX_CHARS,
            top=-1,
            right=+1)

    evaluates to '┕', which has a normal upward leg and bold rightward leg.

    Args:
        first: The character set to use for legs set to -1. If set to None,
            defaults to the same thing as the second character set.
        second: The character set to use for legs set to +1.
        top: Whether the upward leg should be present.
        bottom: Whether the bottom leg should be present.
        left: Whether the left leg should be present.
        right: Whether the right leg should be present.

    Returns:
        A box drawing character approximating the desired properties, or None
        if all legs are set to 0.
    """
    if first is None:
        first = second
    sign = +1
    combo = None
    if first is NORMAL_BOX_CHARS and second is BOLD_BOX_CHARS:
        combo = NORMAL_THEN_BOLD_MIXED_BOX_CHARS
    if first is BOLD_BOX_CHARS and second is NORMAL_BOX_CHARS:
        combo = NORMAL_THEN_BOLD_MIXED_BOX_CHARS
        sign = -1
    if first is NORMAL_BOX_CHARS and second is DOUBLED_BOX_CHARS:
        combo = NORMAL_THEN_DOUBLED_MIXED_BOX_CHARS
    if first is DOUBLED_BOX_CHARS and second is NORMAL_BOX_CHARS:
        combo = NORMAL_THEN_DOUBLED_MIXED_BOX_CHARS
        sign = -1
    if combo is None:
        choice = second if +1 in [top, bottom, left, right] else first
        return choice.char(top=bool(top), bottom=bool(bottom), left=bool(left), right=bool(right))
    return combo.char(top=top * sign, bottom=bottom * sign, left=left * sign, right=right * sign)