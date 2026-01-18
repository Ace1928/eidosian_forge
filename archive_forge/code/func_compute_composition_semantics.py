from nltk.sem.logic import *
def compute_composition_semantics(function, argument):
    assert isinstance(argument, LambdaExpression), '`' + str(argument) + '` must be a lambda expression'
    return LambdaExpression(argument.variable, ApplicationExpression(function, argument.term).simplify())