import re
import pytest
from spacy.attrs import IS_PUNCT, LOWER, ORTH
from spacy.errors import MatchPatternError
from spacy.lang.en import English
from spacy.lang.lex_attrs import LEX_ATTRS
from spacy.matcher import Matcher
from spacy.tokens import Doc, Span, Token
from spacy.vocab import Vocab
@pytest.mark.issue(615)
def test_issue615(en_tokenizer):

    def merge_phrases(matcher, doc, i, matches):
        """Merge a phrase. We have to be careful here because we'll change the
        token indices. To avoid problems, merge all the phrases once we're called
        on the last match."""
        if i != len(matches) - 1:
            return None
        spans = [Span(doc, start, end, label=label) for label, start, end in matches]
        with doc.retokenize() as retokenizer:
            for span in spans:
                tag = 'NNP' if span.label_ else span.root.tag_
                attrs = {'tag': tag, 'lemma': span.text}
                retokenizer.merge(span, attrs=attrs)
                doc.ents = doc.ents + (span,)
    text = 'The golf club is broken'
    pattern = [{'ORTH': 'golf'}, {'ORTH': 'club'}]
    label = 'Sport_Equipment'
    doc = en_tokenizer(text)
    matcher = Matcher(doc.vocab)
    matcher.add(label, [pattern], on_match=merge_phrases)
    matcher(doc)
    entities = list(doc.ents)
    assert entities != []
    assert entities[0].label != 0