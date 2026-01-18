from __future__ import annotations
def assert_offset_equal(offset, base, expected):
    actual = offset + base
    actual_swapped = base + offset
    actual_apply = offset._apply(base)
    try:
        assert actual == expected
        assert actual_swapped == expected
        assert actual_apply == expected
    except AssertionError as err:
        raise AssertionError(f'\nExpected: {expected}\nActual: {actual}\nFor Offset: {offset})\nAt Date: {base}') from err