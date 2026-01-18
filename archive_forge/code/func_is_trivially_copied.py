from pythran.analyses import ImportedIds
from pythran.passmanager import Transformation
import pythran.metadata as metadata
import gast as ast
def is_trivially_copied(node):
    try:
        ast.literal_eval(node)
        return True
    except ValueError:
        pass
    if isinstance(node, (ast.Name, ast.Attribute)):
        return True
    return False