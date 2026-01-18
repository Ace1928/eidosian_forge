from ...language import BaseDefaults, Language
from ...tokens import Doc
from ...util import DummyTokenizer, load_config_from_str, registry
from ...vocab import Vocab
from .lex_attrs import LEX_ATTRS
from .stop_words import STOP_WORDS
@registry.tokenizers('spacy.th.ThaiTokenizer')
def create_thai_tokenizer():

    def thai_tokenizer_factory(nlp):
        return ThaiTokenizer(nlp.vocab)
    return thai_tokenizer_factory