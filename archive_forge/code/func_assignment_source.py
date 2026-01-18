from lib2to3 import fixer_base
from itertools import count
from lib2to3.fixer_util import (Assign, Comma, Call, Newline, Name,
from libfuturize.fixer_util import indentation, suitify, commatize
def assignment_source(num_pre, num_post, LISTNAME, ITERNAME):
    u"""
    Accepts num_pre and num_post, which are counts of values
    before and after the starg (not including the starg)
    Returns a source fit for Assign() from fixer_util
    """
    children = []
    try:
        pre = unicode(num_pre)
        post = unicode(num_post)
    except NameError:
        pre = str(num_pre)
        post = str(num_post)
    if num_pre > 0:
        pre_part = Node(syms.power, [Name(LISTNAME), Node(syms.trailer, [Leaf(token.LSQB, u'['), Node(syms.subscript, [Leaf(token.COLON, u':'), Number(pre)]), Leaf(token.RSQB, u']')])])
        children.append(pre_part)
        children.append(Leaf(token.PLUS, u'+', prefix=u' '))
    main_part = Node(syms.power, [Leaf(token.LSQB, u'[', prefix=u' '), Name(LISTNAME), Node(syms.trailer, [Leaf(token.LSQB, u'['), Node(syms.subscript, [Number(pre) if num_pre > 0 else Leaf(1, u''), Leaf(token.COLON, u':'), Node(syms.factor, [Leaf(token.MINUS, u'-'), Number(post)]) if num_post > 0 else Leaf(1, u'')]), Leaf(token.RSQB, u']'), Leaf(token.RSQB, u']')])])
    children.append(main_part)
    if num_post > 0:
        children.append(Leaf(token.PLUS, u'+', prefix=u' '))
        post_part = Node(syms.power, [Name(LISTNAME, prefix=u' '), Node(syms.trailer, [Leaf(token.LSQB, u'['), Node(syms.subscript, [Node(syms.factor, [Leaf(token.MINUS, u'-'), Number(post)]), Leaf(token.COLON, u':')]), Leaf(token.RSQB, u']')])])
        children.append(post_part)
    source = Node(syms.arith_expr, children)
    return source