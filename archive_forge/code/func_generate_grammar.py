from ast import literal_eval
from typing import TypeVar, Generic, Mapping, Sequence, Set, Union
from parso.pgen2.grammar_parser import GrammarParser, NFAState
def generate_grammar(bnf_grammar: str, token_namespace) -> Grammar:
    """
    ``bnf_text`` is a grammar in extended BNF (using * for repetition, + for
    at-least-once repetition, [] for optional parts, | for alternatives and ()
    for grouping).

    It's not EBNF according to ISO/IEC 14977. It's a dialect Python uses in its
    own parser.
    """
    rule_to_dfas = {}
    start_nonterminal = None
    for nfa_a, nfa_z in GrammarParser(bnf_grammar).parse():
        dfas = _make_dfas(nfa_a, nfa_z)
        _simplify_dfas(dfas)
        rule_to_dfas[nfa_a.from_rule] = dfas
        if start_nonterminal is None:
            start_nonterminal = nfa_a.from_rule
    reserved_strings: Mapping[str, ReservedString] = {}
    for nonterminal, dfas in rule_to_dfas.items():
        for dfa_state in dfas:
            for terminal_or_nonterminal, next_dfa in dfa_state.arcs.items():
                if terminal_or_nonterminal in rule_to_dfas:
                    dfa_state.nonterminal_arcs[terminal_or_nonterminal] = next_dfa
                else:
                    transition = _make_transition(token_namespace, reserved_strings, terminal_or_nonterminal)
                    dfa_state.transitions[transition] = DFAPlan(next_dfa)
    _calculate_tree_traversal(rule_to_dfas)
    return Grammar(start_nonterminal, rule_to_dfas, reserved_strings)