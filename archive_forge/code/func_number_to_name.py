from __future__ import annotations
import typing
def number_to_name(scheme: ColorScheme | ColorSchemeShort, n: int) -> str:
    """
    Return palette name that corresponds to a given number

    Uses alphabetical ordering
    """
    mod = get_palette_module(scheme)
    names = mod.__all__
    if n > len(names):
        raise ValueError(f"There are only '{n}' palettes of type {scheme}. You requested palette no. {n}")
    return names[n - 1]