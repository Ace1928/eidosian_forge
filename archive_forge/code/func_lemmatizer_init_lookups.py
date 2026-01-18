import pytest
from spacy import registry
from spacy.lookups import Lookups
from spacy.util import get_lang_class
@registry.misc('lemmatizer_init_lookups')
def lemmatizer_init_lookups():
    lookups = Lookups()
    lookups.add_table('lemma_lookup', {'cope': 'cope', 'x': 'y'})
    lookups.add_table('lemma_index', {'verb': ('cope', 'cop')})
    lookups.add_table('lemma_exc', {'verb': {'coping': ('cope',)}})
    lookups.add_table('lemma_rules', {'verb': [['ing', '']]})
    return lookups