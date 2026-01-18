from typing import Any, Dict, Iterator
from ...language import BaseDefaults, Language
from ...scorer import Scorer
from ...symbols import POS, X
from ...tokens import Doc
from ...training import validate_examples
from ...util import DummyTokenizer, load_config_from_str, registry
from ...vocab import Vocab
from .lex_attrs import LEX_ATTRS
from .punctuation import TOKENIZER_INFIXES
from .stop_words import STOP_WORDS
from .tag_map import TAG_MAP
def detailed_tokens(self, text: str) -> Iterator[Dict[str, Any]]:
    for node in self.mecab_tokenizer.parse(text, as_nodes=True):
        if node.is_eos():
            break
        surface = node.surface
        feature = node.feature
        tag, _, expr = feature.partition(',')
        lemma, _, remainder = expr.partition('/')
        if lemma == '*':
            lemma = surface
        yield {'surface': surface, 'lemma': lemma, 'tag': tag}