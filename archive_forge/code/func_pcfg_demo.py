import re
from functools import total_ordering
from nltk.featstruct import SLASH, TYPE, FeatDict, FeatStruct, FeatStructReader
from nltk.internals import raise_unorderable_types
from nltk.probability import ImmutableProbabilisticMixIn
from nltk.util import invert_graph, transitive_closure
def pcfg_demo():
    """
    A demonstration showing how a ``PCFG`` can be created and used.
    """
    from nltk import induce_pcfg, treetransforms
    from nltk.corpus import treebank
    from nltk.parse import pchart
    toy_pcfg1 = PCFG.fromstring("\n        S -> NP VP [1.0]\n        NP -> Det N [0.5] | NP PP [0.25] | 'John' [0.1] | 'I' [0.15]\n        Det -> 'the' [0.8] | 'my' [0.2]\n        N -> 'man' [0.5] | 'telescope' [0.5]\n        VP -> VP PP [0.1] | V NP [0.7] | V [0.2]\n        V -> 'ate' [0.35] | 'saw' [0.65]\n        PP -> P NP [1.0]\n        P -> 'with' [0.61] | 'under' [0.39]\n        ")
    toy_pcfg2 = PCFG.fromstring("\n        S    -> NP VP         [1.0]\n        VP   -> V NP          [.59]\n        VP   -> V             [.40]\n        VP   -> VP PP         [.01]\n        NP   -> Det N         [.41]\n        NP   -> Name          [.28]\n        NP   -> NP PP         [.31]\n        PP   -> P NP          [1.0]\n        V    -> 'saw'         [.21]\n        V    -> 'ate'         [.51]\n        V    -> 'ran'         [.28]\n        N    -> 'boy'         [.11]\n        N    -> 'cookie'      [.12]\n        N    -> 'table'       [.13]\n        N    -> 'telescope'   [.14]\n        N    -> 'hill'        [.5]\n        Name -> 'Jack'        [.52]\n        Name -> 'Bob'         [.48]\n        P    -> 'with'        [.61]\n        P    -> 'under'       [.39]\n        Det  -> 'the'         [.41]\n        Det  -> 'a'           [.31]\n        Det  -> 'my'          [.28]\n        ")
    pcfg_prods = toy_pcfg1.productions()
    pcfg_prod = pcfg_prods[2]
    print('A PCFG production:', repr(pcfg_prod))
    print('    pcfg_prod.lhs()  =>', repr(pcfg_prod.lhs()))
    print('    pcfg_prod.rhs()  =>', repr(pcfg_prod.rhs()))
    print('    pcfg_prod.prob() =>', repr(pcfg_prod.prob()))
    print()
    grammar = toy_pcfg2
    print('A PCFG grammar:', repr(grammar))
    print('    grammar.start()       =>', repr(grammar.start()))
    print('    grammar.productions() =>', end=' ')
    print(repr(grammar.productions()).replace(',', ',\n' + ' ' * 26))
    print()
    print('Induce PCFG grammar from treebank data:')
    productions = []
    item = treebank._fileids[0]
    for tree in treebank.parsed_sents(item)[:3]:
        tree.collapse_unary(collapsePOS=False)
        tree.chomsky_normal_form(horzMarkov=2)
        productions += tree.productions()
    S = Nonterminal('S')
    grammar = induce_pcfg(S, productions)
    print(grammar)
    print()
    print('Parse sentence using induced grammar:')
    parser = pchart.InsideChartParser(grammar)
    parser.trace(3)
    sent = treebank.parsed_sents(item)[0].leaves()
    print(sent)
    for parse in parser.parse(sent):
        print(parse)