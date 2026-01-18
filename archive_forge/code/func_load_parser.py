from nltk.data import load
from nltk.grammar import CFG, PCFG, FeatureGrammar
from nltk.parse.chart import Chart, ChartParser
from nltk.parse.featurechart import FeatureChart, FeatureChartParser
from nltk.parse.pchart import InsideChartParser
def load_parser(grammar_url, trace=0, parser=None, chart_class=None, beam_size=0, **load_args):
    """
    Load a grammar from a file, and build a parser based on that grammar.
    The parser depends on the grammar format, and might also depend
    on properties of the grammar itself.

    The following grammar formats are currently supported:
      - ``'cfg'``  (CFGs: ``CFG``)
      - ``'pcfg'`` (probabilistic CFGs: ``PCFG``)
      - ``'fcfg'`` (feature-based CFGs: ``FeatureGrammar``)

    :type grammar_url: str
    :param grammar_url: A URL specifying where the grammar is located.
        The default protocol is ``"nltk:"``, which searches for the file
        in the the NLTK data package.
    :type trace: int
    :param trace: The level of tracing that should be used when
        parsing a text.  ``0`` will generate no tracing output;
        and higher numbers will produce more verbose tracing output.
    :param parser: The class used for parsing; should be ``ChartParser``
        or a subclass.
        If None, the class depends on the grammar format.
    :param chart_class: The class used for storing the chart;
        should be ``Chart`` or a subclass.
        Only used for CFGs and feature CFGs.
        If None, the chart class depends on the grammar format.
    :type beam_size: int
    :param beam_size: The maximum length for the parser's edge queue.
        Only used for probabilistic CFGs.
    :param load_args: Keyword parameters used when loading the grammar.
        See ``data.load`` for more information.
    """
    grammar = load(grammar_url, **load_args)
    if not isinstance(grammar, CFG):
        raise ValueError('The grammar must be a CFG, or a subclass thereof.')
    if isinstance(grammar, PCFG):
        if parser is None:
            parser = InsideChartParser
        return parser(grammar, trace=trace, beam_size=beam_size)
    elif isinstance(grammar, FeatureGrammar):
        if parser is None:
            parser = FeatureChartParser
        if chart_class is None:
            chart_class = FeatureChart
        return parser(grammar, trace=trace, chart_class=chart_class)
    else:
        if parser is None:
            parser = ChartParser
        if chart_class is None:
            chart_class = Chart
        return parser(grammar, trace=trace, chart_class=chart_class)