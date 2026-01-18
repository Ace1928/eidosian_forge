import warnings
from typing import Any, Dict
import srsly
from .. import util
from ..errors import Warnings
from ..language import Language
from ..matcher import Matcher
from ..tokens import Doc
@Language.factory('doc_cleaner', default_config={'attrs': {'tensor': None, '_.trf_data': None}, 'silent': True})
def make_doc_cleaner(nlp: Language, name: str, *, attrs: Dict[str, Any], silent: bool):
    return DocCleaner(attrs, silent=silent)