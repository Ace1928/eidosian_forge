from docutils import nodes, utils
from docutils.parsers.rst import directives
from docutils.parsers.rst.languages import en as _fallback_language_module
from docutils.utils.code_analyzer import Lexer, LexerError
def register_local_role(name, role_fn):
    """
    Register an interpreted text role by its local or language-dependent name.

    :Parameters:
      - `name`: The local or language-dependent name of the interpreted role.
      - `role_fn`: The role function.  See the module docstring.
    """
    set_implicit_options(role_fn)
    _roles[name] = role_fn