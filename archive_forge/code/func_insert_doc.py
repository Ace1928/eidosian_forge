from nipype.interfaces import fsl
from nipype.utils import docparse
import subprocess
from ..interfaces.base import CommandLine
from .misc import is_container
def insert_doc(doc, new_items):
    """Insert ``new_items`` into the beginning of the ``doc``

    Docstrings in ``new_items`` will be inserted right after the
    *Parameters* header but before the existing docs.

    Parameters
    ----------
    doc : str
        The existing docstring we're inserting docmentation into.
    new_items : list
        List of strings to be inserted in the ``doc``.

    Examples
    --------
    >>> from nipype.utils.docparse import insert_doc
    >>> doc = '''Parameters
    ... ----------
    ... outline :
    ...     something about an outline'''

    >>> new_items = ['infile : str', '    The name of the input file']
    >>> new_items.extend(['outfile : str', '    The name of the output file'])
    >>> newdoc = insert_doc(doc, new_items)
    >>> print(newdoc)
    Parameters
    ----------
    infile : str
        The name of the input file
    outfile : str
        The name of the output file
    outline :
        something about an outline

    """
    doclist = doc.split('\n')
    tmpdoc = doclist[:2]
    tmpdoc.extend(new_items)
    tmpdoc.extend(doclist[2:])
    newdoc = []
    for line in tmpdoc:
        newdoc.append(line)
        newdoc.append('\n')
    newdoc.pop(-1)
    return ''.join(newdoc)