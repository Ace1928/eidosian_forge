from __future__ import absolute_import
@composite
def port_numbers(draw, allow_zero=False):
    """
        A strategy which generates port numbers.

        @param allow_zero: Whether to allow port C{0} as a possible value.
        """
    if allow_zero:
        min_value = 0
    else:
        min_value = 1
    return cast(int, draw(integers(min_value=min_value, max_value=65535)))