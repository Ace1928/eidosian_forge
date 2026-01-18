from __future__ import annotations
import itertools
import os
import re
import typing
from functools import lru_cache
from textwrap import dedent, indent
from the `ggplot()`{.py} call is used. If specified, it overrides the \
from showing in the legend. e.g `show_legend={'color': False}`{.py}, \
def document_stat(stat: type[stat]) -> type[stat]:
    """
    Create a structured documentation for the stat

    It replaces `{usage}`, `{common_parameters}` and
    `{aesthetics}` with generated documentation.
    """
    docstring = dedent(stat.__doc__ or '')
    docstring = append_to_section(stat_kwargs, docstring, 'Parameters')
    signature = make_signature(stat.__name__, stat.DEFAULT_PARAMS, common_stat_params, common_stat_param_values)
    usage = STAT_SIGNATURE_TPL.format(signature=signature)
    contents = {f'**{ae}**': '' for ae in sorted(stat.REQUIRED_AES)}
    contents.update(sorted(stat.DEFAULT_AES.items()))
    table = dict_to_table(('Aesthetic', 'Default value'), contents)
    aesthetics_table = AESTHETICS_TABLE_TPL.format(table=table)
    tpl = dedent(stat._aesthetics_doc).strip()
    aesthetics_doc = tpl.replace('{aesthetics_table}', aesthetics_table)
    aesthetics_doc = indent(aesthetics_doc, ' ' * 4)
    d = stat.DEFAULT_PARAMS
    common_parameters = STAT_PARAMS_TPL.format(default_geom=default_class_name(d['geom']), default_position=default_class_name(d['position']), default_na_rm=d['na_rm'], _aesthetics_doc=aesthetics_doc, **common_params_doc).strip()
    docstring = docstring.replace('{usage}', usage)
    docstring = docstring.replace('{common_parameters}', common_parameters)
    stat.__doc__ = docstring
    return stat