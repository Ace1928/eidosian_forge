from nipype.interfaces import fsl
from nipype.utils import docparse
import subprocess
from ..interfaces.base import CommandLine
from .misc import is_container
def build_doc(doc, opts):
    """Build docstring from doc and options

    Parameters
    ----------
    rep_doc : string
        Documentation string
    opts : dict
        Dictionary of option attributes and keys.  Use reverse_opt_map
        to reverse flags and attrs from opt_map class attribute.

    Returns
    -------
    newdoc : string
        The docstring with flags replaced with attribute names and
        formatted to match nipy standards (as best we can).

    """
    doclist = doc.split('\n')
    newdoc = []
    flags_doc = []
    for line in doclist:
        linelist = line.split()
        if not linelist:
            continue
        if ',' in linelist[0]:
            flag = linelist[0].split(',')[0]
        else:
            flag = linelist[0]
        attr = opts.get(flag)
        if attr is not None:
            linelist[0] = '%s :\n    ' % str(attr)
            newline = ' '.join(linelist)
            newdoc.append(newline)
        elif line[0].isspace():
            flags_doc.append(line)
    return format_params(newdoc, flags_doc)