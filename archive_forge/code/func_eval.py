import typing
import uuid
from rpy2.robjects import conversion
from rpy2.robjects.robject import RObject
import rpy2.rinterface as ri
def eval(x: typing.Union[str, ri.ExprSexpVector], envir: typing.Union[None, ri.SexpEnvironment, ri.NULLType, ri.ListSexpVector, ri.PairlistSexpVector, int, ri._MissingArgType]=None, enclos: typing.Union[None, ri.ListSexpVector, ri.PairlistSexpVector, ri.NULLType, ri._MissingArgType]=None) -> RObject:
    """ Evaluate R code. If the input object is an R expression it
    evaluates it directly, if it is a string it parses it before
    evaluating it.

    By default the evaluation is performed in R's global environment
    but a specific environment can be specified.

    This function is a wrapper around rpy2.rinterface.evalr and
    rpy2.rinterface.evalr_expr.

    Args:
    - x (str or ExprSexpVector): a string to be parsed as R code and
    evaluated, or an R expression to be evaluated.
    - envir: An optional R environment where the R code will be
    evaluated.
    - enclos: An optional enclosure.
    Returns:
    The R objects resulting from the evaluation."""
    if envir is None:
        envir = ri.get_evaluation_context()
    if isinstance(x, str):
        res = ri.evalr(x, envir=envir, enclos=enclos)
    else:
        res = ri.evalr_expr(x, envir=envir, enclos=enclos)
    return conversion.rpy2py(res)