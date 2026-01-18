import sys
import logging
import itertools
from pyomo.common.numeric_types import native_types, native_numeric_types
from pyomo.core.base import Constraint, Objective, ComponentMap
import pyomo.core.expr as EXPR
from pyomo.core.expr.numvalue import NumericConstant
from pyomo.core.base.objective import _GeneralObjectiveData, ScalarObjective
from pyomo.core.base import _ExpressionData, Expression
from pyomo.core.base.expression import ScalarExpression, _GeneralExpressionData
from pyomo.core.base.var import ScalarVar, Var, _GeneralVarData, value
from pyomo.core.base.param import ScalarParam, _ParamData
from pyomo.core.kernel.expression import expression, noclone
from pyomo.core.kernel.variable import IVariable, variable
from pyomo.core.kernel.objective import objective
from io import StringIO
def generate_standard_repn(expr, idMap=None, compute_values=True, verbose=False, quadratic=True, repn=None):
    global Results
    if quadratic:
        Results = ResultsWithQuadratics
    else:
        Results = ResultsWithoutQuadratics
    if True:
        if idMap is None:
            idMap = {}
        idMap.setdefault(None, {})
        if repn is None:
            repn = StandardRepn()
        if expr.__class__ in native_numeric_types or not expr.is_potentially_variable():
            if compute_values:
                repn.constant = EXPR.evaluate_expression(expr)
            else:
                repn.constant = expr
            return repn
        elif expr.is_variable_type():
            if expr.fixed:
                if compute_values:
                    repn.constant = value(expr)
                else:
                    repn.constant = expr
                return repn
            repn.linear_coefs = (1,)
            repn.linear_vars = (expr,)
            return repn
        elif expr.__class__ is EXPR.LinearExpression:
            linear_coefs = {}
            linear_vars = {}
            C_ = 0
            if compute_values:
                for arg in expr.args:
                    if arg.__class__ is EXPR.MonomialTermExpression:
                        c, v = arg.args
                        if c.__class__ not in native_numeric_types:
                            c = EXPR.evaluate_expression(c)
                        if v.fixed:
                            C_ += c * v.value
                            continue
                        id_ = id(v)
                        if id_ in linear_coefs:
                            linear_coefs[id_] += c
                        else:
                            linear_coefs[id_] = c
                            linear_vars[id_] = v
                    elif arg.__class__ in native_numeric_types:
                        C_ += arg
                    else:
                        C_ += EXPR.evaluate_expression(arg)
            else:
                for arg in expr.args:
                    if arg.__class__ is EXPR.MonomialTermExpression:
                        c, v = arg.args
                        if v.fixed:
                            C_ += c * v
                            continue
                        id_ = id(v)
                        if id_ in linear_coefs:
                            linear_coefs[id_] += c
                        else:
                            linear_coefs[id_] = c
                            linear_vars[id_] = v
                    else:
                        C_ += arg
            vars_ = []
            coef_ = []
            for id_, coef in linear_coefs.items():
                if coef.__class__ in native_numeric_types and (not coef):
                    continue
                if id_ not in idMap[None]:
                    key = len(idMap) - 1
                    idMap[None][id_] = key
                    idMap[key] = linear_vars[id_]
                else:
                    key = idMap[None][id_]
                vars_.append(idMap[key])
                coef_.append(coef)
            repn.linear_vars = tuple(vars_)
            repn.linear_coefs = tuple(coef_)
            repn.constant = C_
            return repn
        elif not expr.is_expression_type():
            raise ValueError('Unexpected expression type: ' + str(expr))
        return _generate_standard_repn(expr, idMap=idMap, compute_values=compute_values, verbose=verbose, quadratic=quadratic, repn=repn)