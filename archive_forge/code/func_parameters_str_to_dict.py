from __future__ import annotations
import itertools
import os
import re
import typing
from functools import lru_cache
from textwrap import dedent, indent
from the `ggplot()`{.py} call is used. If specified, it overrides the \
from showing in the legend. e.g `show_legend={'color': False}`{.py}, \
def parameters_str_to_dict(param_section: str) -> dict[str, str]:
    """
    Convert a param section to a dict

    Parameters
    ----------
    param_section : str
        Text in the parameter section

    Returns
    -------
    d : dict
        Dictionary of the parameters in the order that they
        are described in the parameters section. The dict
        is of the form `{param: all_parameter_text}`.
        You can reconstruct the `param_section` from the
        keys of the dictionary.

    See Also
    --------
    plotnine.doctools.parameters_dict_to_str
    """
    d = {}
    previous_param = ''
    param_desc: list[str] = []
    for line in param_section.split('\n'):
        param = param_spec(line)
        if param:
            if previous_param:
                d[previous_param] = '\n'.join(param_desc)
            param_desc = [line]
            previous_param = param
        elif param_desc:
            param_desc.append(line)
    if previous_param:
        d[previous_param] = '\n'.join(param_desc)
    return d