import pickle
import pytest
from spacy import registry, util
from spacy.lang.en import English
from spacy.lookups import Lookups
from ..util import make_tempdir
def cope_lookups():
    lookups = Lookups()
    lookups.add_table('lemma_lookup', {'cope': 'cope', 'coped': 'cope'})
    lookups.add_table('lemma_index', {'verb': ('cope', 'cop')})
    lookups.add_table('lemma_exc', {'verb': {'coping': ('cope',)}})
    lookups.add_table('lemma_rules', {'verb': [['ing', '']]})
    return lookups