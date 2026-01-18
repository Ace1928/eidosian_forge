import pytest
from spacy import registry
from spacy.pipeline import DependencyParser
from spacy.pipeline._parser_internals.arc_eager import ArcEager
from spacy.pipeline._parser_internals.nonproj import projectivize
from spacy.pipeline.dep_parser import DEFAULT_PARSER_MODEL
from spacy.tokens import Doc
from spacy.training import Example
from spacy.vocab import Vocab
def test_get_oracle_actions():
    ids, words, tags, heads, deps, ents = ([], [], [], [], [], [])
    for id_, word, tag, head, dep, ent in annot_tuples:
        ids.append(id_)
        words.append(word)
        tags.append(tag)
        heads.append(head)
        deps.append(dep)
        ents.append(ent)
    doc = Doc(Vocab(), words=[t[1] for t in annot_tuples])
    cfg = {'model': DEFAULT_PARSER_MODEL}
    model = registry.resolve(cfg, validate=True)['model']
    parser = DependencyParser(doc.vocab, model)
    parser.moves.add_action(0, '')
    parser.moves.add_action(1, '')
    parser.moves.add_action(1, '')
    parser.moves.add_action(4, 'ROOT')
    heads, deps = projectivize(heads, deps)
    for i, (head, dep) in enumerate(zip(heads, deps)):
        if head > i:
            parser.moves.add_action(2, dep)
        elif head < i:
            parser.moves.add_action(3, dep)
    example = Example.from_dict(doc, {'words': words, 'tags': tags, 'heads': heads, 'deps': deps})
    parser.moves.get_oracle_sequence(example)