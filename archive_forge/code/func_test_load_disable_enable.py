import gc
import numpy
import pytest
from thinc.api import get_current_ops
import spacy
from spacy.lang.en import English
from spacy.lang.en.syntax_iterators import noun_chunks
from spacy.language import Language
from spacy.pipeline import TrainablePipe
from spacy.tokens import Doc
from spacy.training import Example
from spacy.util import SimpleFrozenList, get_arg_names, make_tempdir
from spacy.vocab import Vocab
def test_load_disable_enable():
    """Tests spacy.load() with dis-/enabling components."""
    base_nlp = English()
    for pipe in ('sentencizer', 'tagger', 'parser'):
        base_nlp.add_pipe(pipe)
    with make_tempdir() as tmp_dir:
        base_nlp.to_disk(tmp_dir)
        to_disable = ['parser', 'tagger']
        to_enable = ['tagger', 'parser']
        single_str = 'tagger'
        nlp = spacy.load(tmp_dir, disable=to_disable)
        assert all([comp_name in nlp.disabled for comp_name in to_disable])
        nlp = spacy.load(tmp_dir, enable=to_enable)
        assert all([(comp_name in nlp.disabled) is (comp_name not in to_enable) for comp_name in nlp.component_names])
        nlp = spacy.load(tmp_dir, exclude=single_str)
        assert single_str not in nlp.component_names
        nlp = spacy.load(tmp_dir, disable=single_str)
        assert single_str in nlp.component_names
        assert single_str not in nlp.pipe_names
        assert nlp._disabled == {single_str}
        assert nlp.disabled == [single_str]
        nlp = spacy.load(tmp_dir, enable=to_enable, disable=[comp_name for comp_name in nlp.component_names if comp_name not in to_enable])
        assert all([(comp_name in nlp.disabled) is (comp_name not in to_enable) for comp_name in nlp.component_names])
        with pytest.raises(ValueError):
            spacy.load(tmp_dir, enable=to_enable, disable=['parser'])