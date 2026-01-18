import tempfile
import warnings
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional
import srsly
from ... import util
from ...errors import Errors, Warnings
from ...language import BaseDefaults, Language
from ...scorer import Scorer
from ...tokens import Doc
from ...training import Example, validate_examples
from ...util import DummyTokenizer, load_config_from_str, registry
from ...vocab import Vocab
from .lex_attrs import LEX_ATTRS
from .stop_words import STOP_WORDS
def save_pkuseg_model(path):
    if self.pkuseg_seg:
        if not path.exists():
            path.mkdir(parents=True)
        self.pkuseg_seg.model.save(path)
        self.pkuseg_seg.feature_extractor.save(path)