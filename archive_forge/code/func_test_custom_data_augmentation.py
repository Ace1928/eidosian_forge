import random
from contextlib import contextmanager
import pytest
from spacy.lang.en import English
from spacy.pipeline._parser_internals.nonproj import contains_cycle
from spacy.tokens import Doc, DocBin, Span
from spacy.training import Corpus, Example
from spacy.training.augment import (
from ..util import make_tempdir
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_custom_data_augmentation(nlp, doc):

    def create_spongebob_augmenter(randomize: bool=False):

        def augment(nlp, example):
            text = example.text
            if randomize:
                ch = [c.lower() if random.random() < 0.5 else c.upper() for c in text]
            else:
                ch = [c.lower() if i % 2 else c.upper() for i, c in enumerate(text)]
            example_dict = example.to_dict()
            doc = nlp.make_doc(''.join(ch))
            example_dict['token_annotation']['ORTH'] = [t.text for t in doc]
            yield example
            yield example.from_dict(doc, example_dict)
        return augment
    with make_docbin([doc]) as output_file:
        reader = Corpus(output_file, augmenter=create_spongebob_augmenter())
        corpus = list(reader(nlp))
    orig_text = "Sarah 's sister flew to Silicon Valley via London . "
    augmented = "SaRaH 's sIsTeR FlEw tO SiLiCoN VaLlEy vIa lOnDoN . "
    assert corpus[0].text == orig_text
    assert corpus[0].reference.text == orig_text
    assert corpus[0].predicted.text == orig_text
    assert corpus[1].text == augmented
    assert corpus[1].reference.text == augmented
    assert corpus[1].predicted.text == augmented
    ents = [(e.start, e.end, e.label) for e in doc.ents]
    assert [(e.start, e.end, e.label) for e in corpus[0].reference.ents] == ents
    assert [(e.start, e.end, e.label) for e in corpus[1].reference.ents] == ents