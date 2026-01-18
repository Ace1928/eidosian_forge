from nltk.data import load
from nltk.grammar import CFG, PCFG, FeatureGrammar
from nltk.parse.chart import Chart, ChartParser
from nltk.parse.featurechart import FeatureChart, FeatureChartParser
from nltk.parse.pchart import InsideChartParser
def taggedsent_to_conll(sentence):
    """
    A module to convert a single POS tagged sentence into CONLL format.

    >>> from nltk import word_tokenize, pos_tag
    >>> text = "This is a foobar sentence."
    >>> for line in taggedsent_to_conll(pos_tag(word_tokenize(text))): # doctest: +NORMALIZE_WHITESPACE
    ... 	print(line, end="")
        1	This	_	DT	DT	_	0	a	_	_
        2	is	_	VBZ	VBZ	_	0	a	_	_
        3	a	_	DT	DT	_	0	a	_	_
        4	foobar	_	JJ	JJ	_	0	a	_	_
        5	sentence	_	NN	NN	_	0	a	_	_
        6	.		_	.	.	_	0	a	_	_

    :param sentence: A single input sentence to parse
    :type sentence: list(tuple(str, str))
    :rtype: iter(str)
    :return: a generator yielding a single sentence in CONLL format.
    """
    for i, (word, tag) in enumerate(sentence, start=1):
        input_str = [str(i), word, '_', tag, tag, '_', '0', 'a', '_', '_']
        input_str = '\t'.join(input_str) + '\n'
        yield input_str