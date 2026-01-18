import os
import shutil
import subprocess
import tempfile
from ..printing import latex
from ..kinetics.rates import RadiolyticBase
from ..units import to_unitless, get_derived_unit
def rsys2table(rsys, table_template=None, table_template_dict=None, param_name='Rate constant', **kwargs):
    """
    Renders user provided table_template with table_template_dict which
    also has 'body' entry generated from `rsys2tablines`.

    Defaults is LaTeX table requiring booktabs package to be used
    (add \\usepackage{booktabs} to preamble).

    Parameters
    ----------
    rsys : ReactionSystem
    table_template : string
    table_tempalte_dict : dict used to render table_template (excl. "body")
    param_name : str
        Column header for parameter column
    longtable : bool
        use longtable in defaults. (default: False)
    **kwargs :
        passed onto rsys2tablines

    """
    siunitx = kwargs.pop('siunitx', False)
    line_term = ' \\\\'
    defaults = {'table_env': 'longtable' if kwargs.pop('longtable', False) else 'table', 'alignment': 'llllSll' if siunitx else 'lllllll', 'header': kwargs.get('coldelim', ' & ').join(['Id.', 'Reactants', '', 'Products', '{%s}' % param_name, 'Unit', 'Ref']) + line_term, 'short_cap': rsys.name, 'long_cap': rsys.name, 'label': (rsys.name or 'None').lower()}
    if table_template_dict is None:
        table_template_dict = defaults
    else:
        for k, v in defaults:
            if k not in table_template_dict:
                table_template_dict[k] = v
    if 'body' in table_template_dict:
        raise KeyError("There is already a 'body' key in table_template_dict")
    if 'k_fmt' not in kwargs:
        kwargs['k_fmt'] = '\\num{%.4g}' if siunitx else '%.4g'
    table_template_dict['body'] = (line_term + '\n').join(rsys2tablines(rsys, **kwargs)) + line_term
    if table_template is None:
        if table_template_dict['table_env'] == 'longtable':
            table_template = tex_templates['table']['longtable']
        else:
            table_template = tex_templates['table']['default']
    return table_template % table_template_dict