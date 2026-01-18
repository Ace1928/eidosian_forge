import html.entities
import re
import sys
import typing
from . import __diag__
from .core import *
from .util import (
def nested_expr(opener: Union[str, ParserElement]='(', closer: Union[str, ParserElement]=')', content: typing.Optional[ParserElement]=None, ignore_expr: ParserElement=quoted_string(), *, ignoreExpr: ParserElement=quoted_string()) -> ParserElement:
    """Helper method for defining nested lists enclosed in opening and
    closing delimiters (``"("`` and ``")"`` are the default).

    Parameters:

    - ``opener`` - opening character for a nested list
      (default= ``"("``); can also be a pyparsing expression
    - ``closer`` - closing character for a nested list
      (default= ``")"``); can also be a pyparsing expression
    - ``content`` - expression for items within the nested lists
      (default= ``None``)
    - ``ignore_expr`` - expression for ignoring opening and closing delimiters
      (default= :class:`quoted_string`)
    - ``ignoreExpr`` - this pre-PEP8 argument is retained for compatibility
      but will be removed in a future release

    If an expression is not provided for the content argument, the
    nested expression will capture all whitespace-delimited content
    between delimiters as a list of separate values.

    Use the ``ignore_expr`` argument to define expressions that may
    contain opening or closing characters that should not be treated as
    opening or closing characters for nesting, such as quoted_string or
    a comment expression.  Specify multiple expressions using an
    :class:`Or` or :class:`MatchFirst`. The default is
    :class:`quoted_string`, but if no expressions are to be ignored, then
    pass ``None`` for this argument.

    Example::

        data_type = one_of("void int short long char float double")
        decl_data_type = Combine(data_type + Opt(Word('*')))
        ident = Word(alphas+'_', alphanums+'_')
        number = pyparsing_common.number
        arg = Group(decl_data_type + ident)
        LPAR, RPAR = map(Suppress, "()")

        code_body = nested_expr('{', '}', ignore_expr=(quoted_string | c_style_comment))

        c_function = (decl_data_type("type")
                      + ident("name")
                      + LPAR + Opt(DelimitedList(arg), [])("args") + RPAR
                      + code_body("body"))
        c_function.ignore(c_style_comment)

        source_code = '''
            int is_odd(int x) {
                return (x%2);
            }

            int dec_to_hex(char hchar) {
                if (hchar >= '0' && hchar <= '9') {
                    return (ord(hchar)-ord('0'));
                } else {
                    return (10+ord(hchar)-ord('A'));
                }
            }
        '''
        for func in c_function.search_string(source_code):
            print("%(name)s (%(type)s) args: %(args)s" % func)


    prints::

        is_odd (int) args: [['int', 'x']]
        dec_to_hex (int) args: [['char', 'hchar']]
    """
    if ignoreExpr != ignore_expr:
        ignoreExpr = ignore_expr if ignoreExpr == quoted_string() else ignoreExpr
    if opener == closer:
        raise ValueError('opening and closing strings cannot be the same')
    if content is None:
        if isinstance(opener, str_type) and isinstance(closer, str_type):
            opener = typing.cast(str, opener)
            closer = typing.cast(str, closer)
            if len(opener) == 1 and len(closer) == 1:
                if ignoreExpr is not None:
                    content = Combine(OneOrMore(~ignoreExpr + CharsNotIn(opener + closer + ParserElement.DEFAULT_WHITE_CHARS, exact=1))).set_parse_action(lambda t: t[0].strip())
                else:
                    content = empty.copy() + CharsNotIn(opener + closer + ParserElement.DEFAULT_WHITE_CHARS).set_parse_action(lambda t: t[0].strip())
            elif ignoreExpr is not None:
                content = Combine(OneOrMore(~ignoreExpr + ~Literal(opener) + ~Literal(closer) + CharsNotIn(ParserElement.DEFAULT_WHITE_CHARS, exact=1))).set_parse_action(lambda t: t[0].strip())
            else:
                content = Combine(OneOrMore(~Literal(opener) + ~Literal(closer) + CharsNotIn(ParserElement.DEFAULT_WHITE_CHARS, exact=1))).set_parse_action(lambda t: t[0].strip())
        else:
            raise ValueError('opening and closing arguments must be strings if no content expression is given')
    ret = Forward()
    if ignoreExpr is not None:
        ret <<= Group(Suppress(opener) + ZeroOrMore(ignoreExpr | ret | content) + Suppress(closer))
    else:
        ret <<= Group(Suppress(opener) + ZeroOrMore(ret | content) + Suppress(closer))
    ret.set_name('nested %s%s expression' % (opener, closer))
    return ret