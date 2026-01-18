import jinja2  # type: ignore
from rpy2.robjects import (vectors,
from rpy2 import rinterface
from rpy2.robjects.packages import SourceCode
from rpy2.robjects.packages import wherefrom
from IPython import get_ipython  # type: ignore
def html_sourcecode(sourcecode):
    from pygments import highlight
    from pygments.lexers import SLexer
    from pygments.formatters import HtmlFormatter
    formatter = HtmlFormatter()
    htmlcode = highlight(sourcecode, SLexer(), formatter)
    d = {'sourcecode': htmlcode, 'syntax_highlighting': formatter.get_style_defs()}
    html = template_sourcecode.render(d)
    return html