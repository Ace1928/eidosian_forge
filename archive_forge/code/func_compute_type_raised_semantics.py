from nltk.sem.logic import *
def compute_type_raised_semantics(semantics):
    core = semantics
    parent = None
    while isinstance(core, LambdaExpression):
        parent = core
        core = core.term
    var = Variable('F')
    while var in core.free():
        var = unique_variable(pattern=var)
    core = ApplicationExpression(FunctionVariableExpression(var), core)
    if parent is not None:
        parent.term = core
    else:
        semantics = core
    return LambdaExpression(var, semantics)