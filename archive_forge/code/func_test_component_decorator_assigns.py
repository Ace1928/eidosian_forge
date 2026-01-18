import pytest
from mock import Mock
from spacy.language import Language
from spacy.pipe_analysis import get_attr_info, validate_attrs
def test_component_decorator_assigns():

    @Language.component('c1', assigns=['token.tag', 'doc.tensor'])
    def test_component1(doc):
        return doc

    @Language.component('c2', requires=['token.tag', 'token.pos'], assigns=['token.lemma', 'doc.tensor'])
    def test_component2(doc):
        return doc

    @Language.component('c3', requires=['token.lemma'], assigns=['token._.custom_lemma'])
    def test_component3(doc):
        return doc
    assert Language.has_factory('c1')
    assert Language.has_factory('c2')
    assert Language.has_factory('c3')
    nlp = Language()
    nlp.add_pipe('c1')
    nlp.add_pipe('c2')
    problems = nlp.analyze_pipes()['problems']
    assert problems['c2'] == ['token.pos']
    nlp.add_pipe('c3')
    assert get_attr_info(nlp, 'doc.tensor')['assigns'] == ['c1', 'c2']
    nlp.add_pipe('c1', name='c4')
    test_component4_meta = nlp.get_pipe_meta('c1')
    assert test_component4_meta.factory == 'c1'
    assert nlp.pipe_names == ['c1', 'c2', 'c3', 'c4']
    assert not Language.has_factory('c4')
    assert nlp.pipe_factories['c1'] == 'c1'
    assert nlp.pipe_factories['c4'] == 'c1'
    assert get_attr_info(nlp, 'doc.tensor')['assigns'] == ['c1', 'c2', 'c4']
    assert get_attr_info(nlp, 'token.pos')['requires'] == ['c2']
    assert nlp('hello world')