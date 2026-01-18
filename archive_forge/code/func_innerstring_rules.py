import re
from pygments.lexer import Lexer, RegexLexer, include, bygroups, using, \
from pygments.util import get_bool_opt, shebang_matches
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
from pygments import unistring as uni
def innerstring_rules(ttype):
    return [('%(\\(\\w+\\))?[-#0 +]*([0-9]+|[*])?(\\.([0-9]+|[*]))?[hlL]?[E-GXc-giorsux%]', String.Interpol), ('\\{((\\w+)((\\.\\w+)|(\\[[^\\]]+\\]))*)?(\\![sra])?(\\:(.?[<>=\\^])?[-+ ]?#?0?(\\d+)?,?(\\.\\d+)?[E-GXb-gnosx%]?)?\\}', String.Interpol), ('[^\\\\\\\'"%{\\n]+', ttype), ('[\\\'"\\\\]', ttype), ('%|(\\{{1,2})', ttype)]