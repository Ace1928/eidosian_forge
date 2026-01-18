from __future__ import annotations
import typing
def get_palette_names(scheme: ColorScheme | ColorSchemeShort) -> list[str]:
    """
    Return list of palette names
    """
    mod = get_palette_module(scheme)
    names = mod.__all__
    return names.copy()