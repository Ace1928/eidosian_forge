import os
import shutil
import subprocess
import tempfile
from ..printing import latex
from ..kinetics.rates import RadiolyticBase
from ..units import to_unitless, get_derived_unit
def rsys2pdf_table(rsys, output_dir=None, doc_template=None, doc_template_dict=None, save=True, landscape=False, **kwargs):
    """
    Convenience function to render a ReactionSystem as
    e.g. a pdf using e.g. pdflatex.

    Parameters
    ----------
    rsys : ReactionSystem
    output_dir : path to output directory
        (default: system's temporary folder)
    doc_template : string
        LaTeX boiler plate temlpate including preamble,
        document environment etc.
    doc_template_dict : dict (string -> string)
        dict used to render temlpate (excl. 'table')
    longtable : bool
        use longtable in defaults. (default: False)
    **kwargs :
        passed on to `rsys2table`
    """
    if doc_template is None:
        doc_template = tex_templates['document']['default']
    lscape = ['pdflscape' if landscape == 'pdf' else 'lscape'] if landscape else []
    _pkgs = ['booktabs', 'amsmath', ('pdftex,colorlinks,unicode=True', 'hyperref')] + lscape
    if kwargs.get('longtable', False):
        _pkgs += ['longtable']
    if kwargs.get('siunitx', False):
        _pkgs += ['siunitx']
    _envs = ['tiny'] + (['landscape'] if landscape else [])
    defaults = {'usepkg': '\n'.join([('\\usepackage' + ('[%s]' if isinstance(pkg, tuple) else '') + '{%s}') % pkg for pkg in _pkgs]), 'begins': '\n'.join(['\\begin{%s}' % env for env in _envs]), 'ends': '\n'.join(['\\end{%s}' % env for env in _envs[::-1]])}
    if doc_template_dict is None:
        doc_template_dict = defaults
    else:
        for k, v in defaults:
            if k not in doc_template_dict:
                doc_template_dict[k] = v
    if 'table' in doc_template_dict:
        raise KeyError("There is already a 'table' key in doc_template_dict")
    doc_template_dict['table'] = rsys2table(rsys, **kwargs)
    contents = doc_template % doc_template_dict
    if isinstance(save, str) and save.endswith('.pdf'):
        texfname = save.rstrip('.pdf') + '.tex'
        pdffname = save
    else:
        texfname = 'output.tex'
        pdffname = 'output.pdf'
    return render_tex_to_pdf(contents, texfname, pdffname, output_dir, save)