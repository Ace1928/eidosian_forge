import warnings
from typing import Any, Dict
import srsly
from .. import util
from ..errors import Warnings
from ..language import Language
from ..matcher import Matcher
from ..tokens import Doc
@Language.component('merge_noun_chunks', requires=['token.dep', 'token.tag', 'token.pos'], retokenizes=True)
def merge_noun_chunks(doc: Doc) -> Doc:
    """Merge noun chunks into a single token.

    doc (Doc): The Doc object.
    RETURNS (Doc): The Doc object with merged noun chunks.

    DOCS: https://spacy.io/api/pipeline-functions#merge_noun_chunks
    """
    if not doc.has_annotation('DEP'):
        return doc
    with doc.retokenize() as retokenizer:
        for np in doc.noun_chunks:
            attrs = {'tag': np.root.tag, 'dep': np.root.dep}
            retokenizer.merge(np, attrs=attrs)
    return doc