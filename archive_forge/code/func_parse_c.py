from sympy.external import import_module
import os
def parse_c(source):
    """Function for converting a C source code

    The function reads the source code present in the given file and parses it
    to give out SymPy Expressions

    Returns
    =======

    src : list
        List of Python expression strings

    """
    converter = CCodeConverter()
    if os.path.exists(source):
        src = converter.parse(source, flags=[])
    else:
        src = converter.parse_str(source, flags=[])
    return src