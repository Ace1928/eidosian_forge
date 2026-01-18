import re
from pygments.lexers import (
from pygments.lexer import (
from pygments.token import (
from pygments.util import get_bool_opt
def build_ipy_lexer(python3):
    """Builds IPython lexers depending on the value of `python3`.

    The lexer inherits from an appropriate Python lexer and then adds
    information about IPython specific keywords (i.e. magic commands,
    shell commands, etc.)

    Parameters
    ----------
    python3 : bool
        If `True`, then build an IPython lexer from a Python 3 lexer.

    """
    if python3:
        PyLexer = Python3Lexer
        name = 'IPython3'
        aliases = ['ipython3']
        doc = 'IPython3 Lexer'
    else:
        PyLexer = PythonLexer
        name = 'IPython'
        aliases = ['ipython2', 'ipython']
        doc = 'IPython Lexer'
    ipython_tokens = [('(?s)(\\s*)(%%capture)([^\\n]*\\n)(.*)', bygroups(Text, Operator, Text, using(PyLexer))), ('(?s)(\\s*)(%%debug)([^\\n]*\\n)(.*)', bygroups(Text, Operator, Text, using(PyLexer))), ('(?is)(\\s*)(%%html)([^\\n]*\\n)(.*)', bygroups(Text, Operator, Text, using(HtmlLexer))), ('(?s)(\\s*)(%%javascript)([^\\n]*\\n)(.*)', bygroups(Text, Operator, Text, using(JavascriptLexer))), ('(?s)(\\s*)(%%js)([^\\n]*\\n)(.*)', bygroups(Text, Operator, Text, using(JavascriptLexer))), ('(?s)(\\s*)(%%latex)([^\\n]*\\n)(.*)', bygroups(Text, Operator, Text, using(TexLexer))), ('(?s)(\\s*)(%%perl)([^\\n]*\\n)(.*)', bygroups(Text, Operator, Text, using(PerlLexer))), ('(?s)(\\s*)(%%prun)([^\\n]*\\n)(.*)', bygroups(Text, Operator, Text, using(PyLexer))), ('(?s)(\\s*)(%%pypy)([^\\n]*\\n)(.*)', bygroups(Text, Operator, Text, using(PyLexer))), ('(?s)(\\s*)(%%python)([^\\n]*\\n)(.*)', bygroups(Text, Operator, Text, using(PyLexer))), ('(?s)(\\s*)(%%python2)([^\\n]*\\n)(.*)', bygroups(Text, Operator, Text, using(PythonLexer))), ('(?s)(\\s*)(%%python3)([^\\n]*\\n)(.*)', bygroups(Text, Operator, Text, using(Python3Lexer))), ('(?s)(\\s*)(%%ruby)([^\\n]*\\n)(.*)', bygroups(Text, Operator, Text, using(RubyLexer))), ('(?s)(\\s*)(%%time)([^\\n]*\\n)(.*)', bygroups(Text, Operator, Text, using(PyLexer))), ('(?s)(\\s*)(%%timeit)([^\\n]*\\n)(.*)', bygroups(Text, Operator, Text, using(PyLexer))), ('(?s)(\\s*)(%%writefile)([^\\n]*\\n)(.*)', bygroups(Text, Operator, Text, using(PyLexer))), ('(?s)(\\s*)(%%file)([^\\n]*\\n)(.*)', bygroups(Text, Operator, Text, using(PyLexer))), ('(?s)(\\s*)(%%)(\\w+)(.*)', bygroups(Text, Operator, Keyword, Text)), ('(?s)(^\\s*)(%%!)([^\\n]*\\n)(.*)', bygroups(Text, Operator, Text, using(BashLexer))), ('(%%?)(\\w+)(\\?\\??)$', bygroups(Operator, Keyword, Operator)), ('\\b(\\?\\??)(\\s*)$', bygroups(Operator, Text)), ('(%)(sx|sc|system)(.*)(\\n)', bygroups(Operator, Keyword, using(BashLexer), Text)), ('(%)(\\w+)(.*\\n)', bygroups(Operator, Keyword, Text)), ('^(!!)(.+)(\\n)', bygroups(Operator, using(BashLexer), Text)), ('(!)(?!=)(.+)(\\n)', bygroups(Operator, using(BashLexer), Text)), ('^(\\s*)(\\?\\??)(\\s*%{0,2}[\\w\\.\\*]*)', bygroups(Text, Operator, Text)), ('(\\s*%{0,2}[\\w\\.\\*]*)(\\?\\??)(\\s*)$', bygroups(Text, Operator, Text))]
    tokens = PyLexer.tokens.copy()
    tokens['root'] = ipython_tokens + tokens['root']
    attrs = {'name': name, 'aliases': aliases, 'filenames': [], '__doc__': doc, 'tokens': tokens}
    return type(name, (PyLexer,), attrs)