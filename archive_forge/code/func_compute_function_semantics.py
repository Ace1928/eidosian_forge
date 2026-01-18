from nltk.sem.logic import *
def compute_function_semantics(function, argument):
    return ApplicationExpression(function, argument).simplify()