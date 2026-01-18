import ast
import re
from collections import OrderedDict
def parse_ascconv(ascconv_str, str_delim='"'):
    """Parse the 'ASCCONV' format from `input_str`.

    Parameters
    ----------
    ascconv_str : str
        The string we are parsing
    str_delim : str, optional
        String delimiter.  Typically '"' or '""'

    Returns
    -------
    prot_dict : OrderedDict
        Meta data pulled from the ASCCONV section.
    attrs : OrderedDict
        Any attributes stored in the 'ASCCONV BEGIN' line

    Raises
    ------
    AsconvParseError
        A line of the ASCCONV section could not be parsed.
    """
    attrs, content = ASCCONV_RE.match(ascconv_str).groups()
    attrs = OrderedDict((tuple(x.split('=')) for x in attrs.split()))
    content = content.replace(str_delim, '"""').replace('\\', '\\\\')
    tree = ast.parse(content)
    prot_dict = OrderedDict()
    for assign in tree.body:
        atoms = assign2atoms(assign)
        obj_to_index, key = obj_from_atoms(atoms, prot_dict)
        obj_to_index[key] = _get_value(assign)
    return (prot_dict, attrs)