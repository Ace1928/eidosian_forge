from nltk.parse import load_parser
from nltk.parse.featurechart import InstantiateVarsChart
from nltk.sem.logic import ApplicationExpression, LambdaExpression, Variable
def parse_with_bindops(sentence, grammar=None, trace=0):
    """
    Use a grammar with Binding Operators to parse a sentence.
    """
    if not grammar:
        grammar = 'grammars/book_grammars/storage.fcfg'
    parser = load_parser(grammar, trace=trace, chart_class=InstantiateVarsChart)
    tokens = sentence.split()
    return list(parser.parse(tokens))