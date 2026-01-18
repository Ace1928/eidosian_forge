from typing import List
from thinc.api import Model
from thinc.types import Floats2d
from spacy.tokens import Doc
from spacy.util import registry
def lazy_init_tok2vec_init(model: Model, X=None, Y=None):
    width = model.attrs['width']
    model.set_dim('nO', width)