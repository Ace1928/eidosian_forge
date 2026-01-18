from nltk.data import load
from nltk.grammar import CFG, PCFG, FeatureGrammar
from nltk.parse.chart import Chart, ChartParser
from nltk.parse.featurechart import FeatureChart, FeatureChartParser
from nltk.parse.pchart import InsideChartParser
def taggedsents_to_conll(sentences):
    """
    A module to convert the a POS tagged document stream
    (i.e. list of list of tuples, a list of sentences) and yield lines
    in CONLL format. This module yields one line per word and two newlines
    for end of sentence.

    >>> from nltk import word_tokenize, sent_tokenize, pos_tag
    >>> text = "This is a foobar sentence. Is that right?"
    >>> sentences = [pos_tag(word_tokenize(sent)) for sent in sent_tokenize(text)]
    >>> for line in taggedsents_to_conll(sentences): # doctest: +NORMALIZE_WHITESPACE
    ...     if line:
    ...         print(line, end="")
    1	This	_	DT	DT	_	0	a	_	_
    2	is	_	VBZ	VBZ	_	0	a	_	_
    3	a	_	DT	DT	_	0	a	_	_
    4	foobar	_	JJ	JJ	_	0	a	_	_
    5	sentence	_	NN	NN	_	0	a	_	_
    6	.		_	.	.	_	0	a	_	_
    <BLANKLINE>
    <BLANKLINE>
    1	Is	_	VBZ	VBZ	_	0	a	_	_
    2	that	_	IN	IN	_	0	a	_	_
    3	right	_	NN	NN	_	0	a	_	_
    4	?	_	.	.	_	0	a	_	_
    <BLANKLINE>
    <BLANKLINE>

    :param sentences: Input sentences to parse
    :type sentence: list(list(tuple(str, str)))
    :rtype: iter(str)
    :return: a generator yielding sentences in CONLL format.
    """
    for sentence in sentences:
        yield from taggedsent_to_conll(sentence)
        yield '\n\n'