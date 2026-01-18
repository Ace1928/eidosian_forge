import pytest
from mock import Mock
from spacy.language import Language
from spacy.pipe_analysis import get_attr_info, validate_attrs
def test_component_factories_class_func():
    """Test that class components can implement a from_nlp classmethod that
    gives them access to the nlp object and config via the factory."""

    class TestComponent5:

        def __call__(self, doc):
            return doc
    mock = Mock()
    mock.return_value = TestComponent5()

    def test_componen5_factory(nlp, foo: str='bar', name='c5'):
        return mock(nlp, foo=foo)
    Language.factory('c5', func=test_componen5_factory)
    assert Language.has_factory('c5')
    nlp = Language()
    nlp.add_pipe('c5', config={'foo': 'bar'})
    assert nlp('hello world')
    mock.assert_called_once_with(nlp, foo='bar')