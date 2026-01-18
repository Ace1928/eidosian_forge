from textwrap import dedent
import numpy as np
import pytest
from pandas import (
from pandas.io.formats.style import Styler
from pandas.io.formats.style_render import (
@pytest.mark.parametrize('environment', ['table', 'figure*', None])
def test_comprehensive(df_ext, environment):
    cidx = MultiIndex.from_tuples([('Z', 'a'), ('Z', 'b'), ('Y', 'c')])
    ridx = MultiIndex.from_tuples([('A', 'a'), ('A', 'b'), ('B', 'c')])
    df_ext.index, df_ext.columns = (ridx, cidx)
    stlr = df_ext.style
    stlr.set_caption('mycap')
    stlr.set_table_styles([{'selector': 'label', 'props': ':{fig§item}'}, {'selector': 'position', 'props': ':h!'}, {'selector': 'position_float', 'props': ':centering'}, {'selector': 'column_format', 'props': ':rlrlr'}, {'selector': 'toprule', 'props': ':toprule'}, {'selector': 'midrule', 'props': ':midrule'}, {'selector': 'bottomrule', 'props': ':bottomrule'}, {'selector': 'rowcolors', 'props': ':{3}{pink}{}'}])
    stlr.highlight_max(axis=0, props='textbf:--rwrap;cellcolor:[rgb]{1,1,0.6}--rwrap')
    stlr.highlight_max(axis=None, props='Huge:--wrap;', subset=[('Z', 'a'), ('Z', 'b')])
    expected = '\\begin{table}[h!]\n\\centering\n\\caption{mycap}\n\\label{fig:item}\n\\rowcolors{3}{pink}{}\n\\begin{tabular}{rlrlr}\n\\toprule\n &  & \\multicolumn{2}{r}{Z} & Y \\\\\n &  & a & b & c \\\\\n\\midrule\n\\multirow[c]{2}{*}{A} & a & 0 & \\textbf{\\cellcolor[rgb]{1,1,0.6}{-0.61}} & ab \\\\\n & b & 1 & -1.22 & cd \\\\\nB & c & \\textbf{\\cellcolor[rgb]{1,1,0.6}{{\\Huge 2}}} & -2.22 & \\textbf{\\cellcolor[rgb]{1,1,0.6}{de}} \\\\\n\\bottomrule\n\\end{tabular}\n\\end{table}\n'.replace('table', environment if environment else 'table')
    result = stlr.format(precision=2).to_latex(environment=environment)
    assert result == expected