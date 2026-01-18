import pytest
from spacy import registry
from spacy.pipeline import DependencyParser
from spacy.pipeline._parser_internals.arc_eager import ArcEager
from spacy.pipeline._parser_internals.nonproj import projectivize
from spacy.pipeline.dep_parser import DEFAULT_PARSER_MODEL
from spacy.tokens import Doc
from spacy.training import Example
from spacy.vocab import Vocab
def get_sequence_costs(M, words, heads, deps, transitions):
    doc = Doc(Vocab(), words=words)
    example = Example.from_dict(doc, {'heads': heads, 'deps': deps})
    states, golds, _ = M.init_gold_batch([example])
    state = states[0]
    gold = golds[0]
    cost_history = []
    for gold_action in transitions:
        gold.update(state)
        state_costs = {}
        for i in range(M.n_moves):
            name = M.class_name(i)
            state_costs[name] = M.get_cost(state, gold, i)
        M.transition(state, gold_action)
        cost_history.append(state_costs)
    return (state, cost_history)