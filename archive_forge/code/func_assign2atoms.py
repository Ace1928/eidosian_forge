import ast
import re
from collections import OrderedDict
def assign2atoms(assign_ast, default_class=int):
    """Parse single assignment ast from ascconv line into atoms

    Parameters
    ----------
    assign_ast : assignment statement ast
        ast derived from single line of ascconv file.
    default_class : class, optional
        Class that will create an object where we cannot yet know the object
        type in the assignment.

    Returns
    -------
    atoms : list
        List of :class:`atoms`.  See docstring for :class:`atoms`.  Defines
        left to right sequence of assignment in `line_ast`.
    """
    if not len(assign_ast.targets) == 1:
        raise AscconvParseError('Too many targets in assign')
    target = assign_ast.targets[0]
    atoms = []
    prev_target_type = default_class
    while True:
        if isinstance(target, ast.Name):
            atoms.append(Atom(target, prev_target_type, target.id))
            break
        if isinstance(target, ast.Attribute):
            atoms.append(Atom(target, prev_target_type, target.attr))
            target = target.value
            prev_target_type = OrderedDict
        elif isinstance(target, ast.Subscript):
            if isinstance(target.slice, ast.Constant):
                index = target.slice.n
            else:
                index = target.slice.value.n
            atoms.append(Atom(target, prev_target_type, index))
            target = target.value
            prev_target_type = list
        else:
            raise AscconvParseError(f'Unexpected LHS element {target}')
    return reversed(atoms)