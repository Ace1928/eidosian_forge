import random
import numpy
import pytest
import srsly
from thinc.api import Adam, compounding
import spacy
from spacy.lang.en import English
from spacy.tokens import Doc, DocBin
from spacy.training import (
from spacy.training.align import get_alignments
from spacy.training.alignment_array import AlignmentArray
from spacy.training.converters import json_to_docs
from spacy.training.loop import train_while_improving
from spacy.util import (
from ..util import make_tempdir
@pytest.mark.issue(999)
def test_issue999():
    """Test that adding entities and resuming training works passably OK.
    There are two issues here:
    1) We have to re-add labels. This isn't very nice.
    2) There's no way to set the learning rate for the weight update, so we
        end up out-of-scale, causing it to learn too fast.
    """
    TRAIN_DATA = [['hey', []], ['howdy', []], ['hey there', []], ['hello', []], ['hi', []], ["i'm looking for a place to eat", []], ["i'm looking for a place in the north of town", [(31, 36, 'LOCATION')]], ['show me chinese restaurants', [(8, 15, 'CUISINE')]], ['show me chines restaurants', [(8, 14, 'CUISINE')]]]
    nlp = English()
    ner = nlp.add_pipe('ner')
    for _, offsets in TRAIN_DATA:
        for start, end, label in offsets:
            ner.add_label(label)
    nlp.initialize()
    for itn in range(20):
        random.shuffle(TRAIN_DATA)
        for raw_text, entity_offsets in TRAIN_DATA:
            example = Example.from_dict(nlp.make_doc(raw_text), {'entities': entity_offsets})
            nlp.update([example])
    with make_tempdir() as model_dir:
        nlp.to_disk(model_dir)
        nlp2 = load_model_from_path(model_dir)
    for raw_text, entity_offsets in TRAIN_DATA:
        doc = nlp2(raw_text)
        ents = {(ent.start_char, ent.end_char): ent.label_ for ent in doc.ents}
        for start, end, label in entity_offsets:
            if (start, end) in ents:
                assert ents[start, end] == label
                break
            elif entity_offsets:
                raise Exception(ents)