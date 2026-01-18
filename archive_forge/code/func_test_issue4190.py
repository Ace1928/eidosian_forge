import pickle
import re
import pytest
from spacy.attrs import ENT_IOB, ENT_TYPE
from spacy.lang.en import English
from spacy.tokenizer import Tokenizer
from spacy.tokens import Doc
from spacy.util import (
from ..util import assert_packed_msg_equal, make_tempdir
@pytest.mark.issue(4190)
def test_issue4190():

    def customize_tokenizer(nlp):
        prefix_re = compile_prefix_regex(nlp.Defaults.prefixes)
        suffix_re = compile_suffix_regex(nlp.Defaults.suffixes)
        infix_re = compile_infix_regex(nlp.Defaults.infixes)
        exceptions = {k: v for k, v in dict(nlp.Defaults.tokenizer_exceptions).items() if not (len(k) == 2 and k[1] == '.')}
        new_tokenizer = Tokenizer(nlp.vocab, exceptions, prefix_search=prefix_re.search, suffix_search=suffix_re.search, infix_finditer=infix_re.finditer, token_match=nlp.tokenizer.token_match, faster_heuristics=False)
        nlp.tokenizer = new_tokenizer
    test_string = 'Test c.'
    nlp_1 = English()
    doc_1a = nlp_1(test_string)
    result_1a = [token.text for token in doc_1a]
    customize_tokenizer(nlp_1)
    doc_1b = nlp_1(test_string)
    result_1b = [token.text for token in doc_1b]
    with make_tempdir() as model_dir:
        nlp_1.to_disk(model_dir)
        nlp_2 = load_model(model_dir)
    doc_2 = nlp_2(test_string)
    result_2 = [token.text for token in doc_2]
    assert result_1b == result_2
    assert nlp_2.tokenizer.faster_heuristics is False