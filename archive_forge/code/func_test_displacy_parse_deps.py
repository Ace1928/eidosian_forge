import numpy
import pytest
from spacy import displacy
from spacy.displacy.render import DependencyRenderer, EntityRenderer, SpanRenderer
from spacy.lang.en import English
from spacy.lang.fa import Persian
from spacy.tokens import Doc, Span
def test_displacy_parse_deps(en_vocab):
    """Test that deps and tags on a Doc are converted into displaCy's format."""
    words = ['This', 'is', 'a', 'sentence']
    heads = [1, 1, 3, 1]
    pos = ['DET', 'VERB', 'DET', 'NOUN']
    tags = ['DT', 'VBZ', 'DT', 'NN']
    deps = ['nsubj', 'ROOT', 'det', 'attr']
    doc = Doc(en_vocab, words=words, heads=heads, pos=pos, tags=tags, deps=deps)
    deps = displacy.parse_deps(doc)
    assert isinstance(deps, dict)
    assert deps['words'] == [{'lemma': None, 'text': words[0], 'tag': pos[0]}, {'lemma': None, 'text': words[1], 'tag': pos[1]}, {'lemma': None, 'text': words[2], 'tag': pos[2]}, {'lemma': None, 'text': words[3], 'tag': pos[3]}]
    assert deps['arcs'] == [{'start': 0, 'end': 1, 'label': 'nsubj', 'dir': 'left'}, {'start': 2, 'end': 3, 'label': 'det', 'dir': 'left'}, {'start': 1, 'end': 3, 'label': 'attr', 'dir': 'right'}]
    deps = displacy.parse_deps(doc[:])
    assert isinstance(deps, dict)
    assert deps['words'] == [{'lemma': None, 'text': words[0], 'tag': pos[0]}, {'lemma': None, 'text': words[1], 'tag': pos[1]}, {'lemma': None, 'text': words[2], 'tag': pos[2]}, {'lemma': None, 'text': words[3], 'tag': pos[3]}]
    assert deps['arcs'] == [{'start': 0, 'end': 1, 'label': 'nsubj', 'dir': 'left'}, {'start': 2, 'end': 3, 'label': 'det', 'dir': 'left'}, {'start': 1, 'end': 3, 'label': 'attr', 'dir': 'right'}]