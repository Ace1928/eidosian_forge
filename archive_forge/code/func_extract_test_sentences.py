from nltk.data import load
from nltk.grammar import CFG, PCFG, FeatureGrammar
from nltk.parse.chart import Chart, ChartParser
from nltk.parse.featurechart import FeatureChart, FeatureChartParser
from nltk.parse.pchart import InsideChartParser
def extract_test_sentences(string, comment_chars='#%;', encoding=None):
    """
    Parses a string with one test sentence per line.
    Lines can optionally begin with:

    - a bool, saying if the sentence is grammatical or not, or
    - an int, giving the number of parse trees is should have,

    The result information is followed by a colon, and then the sentence.
    Empty lines and lines beginning with a comment char are ignored.

    :return: a list of tuple of sentences and expected results,
        where a sentence is a list of str,
        and a result is None, or bool, or int

    :param comment_chars: ``str`` of possible comment characters.
    :param encoding: the encoding of the string, if it is binary
    """
    if encoding is not None:
        string = string.decode(encoding)
    sentences = []
    for sentence in string.split('\n'):
        if sentence == '' or sentence[0] in comment_chars:
            continue
        split_info = sentence.split(':', 1)
        result = None
        if len(split_info) == 2:
            if split_info[0] in ['True', 'true', 'False', 'false']:
                result = split_info[0] in ['True', 'true']
                sentence = split_info[1]
            else:
                result = int(split_info[0])
                sentence = split_info[1]
        tokens = sentence.split()
        if tokens == []:
            continue
        sentences += [(tokens, result)]
    return sentences