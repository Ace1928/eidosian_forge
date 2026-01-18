import pytest
from thinc.api import ConfigValidationError, Linear, Model
import spacy
from spacy.lang.de import German
from spacy.lang.en import English
from spacy.language import Language
from spacy.pipeline.tok2vec import DEFAULT_TOK2VEC_MODEL
from spacy.tokens import Doc
from spacy.util import SimpleFrozenDict, combine_score_weights, registry
from ..util import make_tempdir
@pytest.mark.issue(5137)
def test_issue5137():
    factory_name = 'test_issue5137'
    pipe_name = 'my_component'

    @Language.factory(factory_name)
    class MyComponent:

        def __init__(self, nlp, name=pipe_name, categories='all_categories'):
            self.nlp = nlp
            self.categories = categories
            self.name = name

        def __call__(self, doc):
            pass

        def to_disk(self, path, **kwargs):
            pass

        def from_disk(self, path, **cfg):
            pass
    nlp = English()
    my_component = nlp.add_pipe(factory_name, name=pipe_name)
    assert my_component.categories == 'all_categories'
    with make_tempdir() as tmpdir:
        nlp.to_disk(tmpdir)
        overrides = {'components': {pipe_name: {'categories': 'my_categories'}}}
        nlp2 = spacy.load(tmpdir, config=overrides)
        assert nlp2.get_pipe(pipe_name).categories == 'my_categories'