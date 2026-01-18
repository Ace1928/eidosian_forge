from pythran.analyses import Aliases
from pythran.passmanager import Transformation
from pythran.syntax import PythranSyntaxError
from pythran.tables import MODULES
import gast as ast
from copy import deepcopy
def handle_keywords(self, func, node, offset=0):
    """
        Gather keywords to positional argument information

        Assumes the named parameter exist, raises a KeyError otherwise
        """
    func_argument_names = {}
    for i, arg in enumerate(func.args.args[offset:]):
        assert isinstance(arg, ast.Name)
        func_argument_names[arg.id] = i
    func_argument_kwonly_names = {}
    for i, arg in enumerate(func.args.kwonlyargs):
        assert isinstance(arg, ast.Name)
        func_argument_kwonly_names[arg.id] = i
    nargs = len(func.args.args) - offset
    defaults = func.args.defaults
    keywords = {func_argument_names[kw.arg]: kw.value for kw in node.keywords if kw.arg not in func_argument_kwonly_names}
    keywords_only = []
    nb_kw = len(node.keywords)
    for i, kw in enumerate(list(reversed(node.keywords))):
        if kw.arg in func_argument_kwonly_names:
            keywords_only.append((func_argument_kwonly_names[kw.arg], kw.value))
            node.keywords.pop(nb_kw - i - 1)
    keywords_only = [v for _, v in sorted(keywords_only)]
    extra_keyword_offset = max(keywords.keys()) if keywords else 0
    node.args.extend([None] * (1 + extra_keyword_offset - len(node.args)))
    replacements = {}
    for index, arg in enumerate(node.args):
        if arg is None:
            if index in keywords:
                replacements[index] = deepcopy(keywords[index])
            else:
                replacements[index] = deepcopy(defaults[index - nargs])
    if not keywords_only:
        return replacements
    node.args.append(ast.Call(ast.Attribute(ast.Attribute(ast.Name('builtins', ast.Load(), None, None), 'pythran', ast.Load()), 'kwonly', ast.Load()), [], []))
    node.args.extend(keywords_only)
    return replacements