import itertools
import logging
import sys
import builtins
from contextlib import nullcontext
from pyomo.common.errors import TemplateExpressionError
from pyomo.core.expr.base import ExpressionBase, ExpressionArgs_Mixin, NPV_Mixin
from pyomo.core.expr.logical_expr import BooleanExpression
from pyomo.core.expr.numeric_expr import (
from pyomo.core.expr.numvalue import (
from pyomo.core.expr.relational_expr import tuple_to_relational_expr
from pyomo.core.expr.visitor import (
def templatize_rule(block, rule, index_set):
    import pyomo.core.base.set
    context = _template_iter_context()
    internal_error = None
    try:
        with _TemplateIterManager.init(context, pyomo.core.base.set._FiniteSetMixin, GetItemExpression, GetAttrExpression):
            if index_set is not None:
                indices = next(iter(context.get_iter(index_set)))
                try:
                    context.cache.pop()
                except IndexError:
                    assert indices is None
                    indices = ()
            else:
                indices = ()
            if type(indices) is not tuple:
                indices = (indices,)
            return (rule(block, indices), indices)
    except:
        internal_error = sys.exc_info()
        raise
    finally:
        if len(context.cache):
            if internal_error is not None:
                logger.error("The following exception was raised when templatizing the rule '%s':\n\t%s" % (rule.name, internal_error[1]))
            raise TemplateExpressionError(None, 'Explicit iteration (for loops) over Sets is not supported by template expressions.  Encountered loop over %s' % (context.cache[-1][0]._set,))
    return (None, indices)