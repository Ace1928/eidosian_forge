import itertools
import re
import warnings
from functools import total_ordering
from nltk.grammar import PCFG, is_nonterminal, is_terminal
from nltk.internals import raise_unorderable_types
from nltk.parse.api import ParserI
from nltk.tree import Tree
from nltk.util import OrderedDict
def chart_parse(self, tokens, trace=None):
    """
        Return the final parse ``Chart`` from which all possible
        parse trees can be extracted.

        :param tokens: The sentence to be parsed
        :type tokens: list(str)
        :rtype: Chart
        """
    if trace is None:
        trace = self._trace
    trace_new_edges = self._trace_new_edges
    tokens = list(tokens)
    self._grammar.check_coverage(tokens)
    chart = self._chart_class(tokens)
    grammar = self._grammar
    trace_edge_width = self._trace_chart_width // (chart.num_leaves() + 1)
    if trace:
        print(chart.pretty_format_leaves(trace_edge_width))
    if self._use_agenda:
        for axiom in self._axioms:
            new_edges = list(axiom.apply(chart, grammar))
            trace_new_edges(chart, axiom, new_edges, trace, trace_edge_width)
        inference_rules = self._inference_rules
        agenda = chart.edges()
        agenda.reverse()
        while agenda:
            edge = agenda.pop()
            for rule in inference_rules:
                new_edges = list(rule.apply(chart, grammar, edge))
                if trace:
                    trace_new_edges(chart, rule, new_edges, trace, trace_edge_width)
                agenda += new_edges
    else:
        edges_added = True
        while edges_added:
            edges_added = False
            for rule in self._strategy:
                new_edges = list(rule.apply_everywhere(chart, grammar))
                edges_added = len(new_edges)
                trace_new_edges(chart, rule, new_edges, trace, trace_edge_width)
    return chart