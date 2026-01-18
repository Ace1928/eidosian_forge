import ctypes
import os
from pathlib import Path
import pytest
from thinc.api import (
from thinc.compat import has_cupy_gpu, has_torch_mps_gpu
from spacy import prefer_gpu, require_cpu, require_gpu, util
from spacy.about import __version__ as spacy_version
from spacy.lang.en import English
from spacy.lang.nl import Dutch
from spacy.language import DEFAULT_CONFIG_PATH
from spacy.ml._precomputable_affine import (
from spacy.schemas import ConfigSchemaTraining, TokenPattern, TokenPatternSchema
from spacy.training.batchers import minibatch_by_words
from spacy.util import (
from .util import get_random_doc, make_tempdir
from spacy import Language
def test_import_code():
    code_str = '\nfrom spacy import Language\n\nclass DummyComponent:\n    def __init__(self, vocab, name):\n        pass\n\n    def initialize(self, get_examples, *, nlp, dummy_param: int):\n        pass\n\n@Language.factory(\n    "dummy_component",\n)\ndef make_dummy_component(\n    nlp: Language, name: str\n):\n    return DummyComponent(nlp.vocab, name)\n'
    with make_tempdir() as temp_dir:
        code_path = os.path.join(temp_dir, 'code.py')
        with open(code_path, 'w') as fileh:
            fileh.write(code_str)
        import_file('python_code', code_path)
        config = {'initialize': {'components': {'dummy_component': {'dummy_param': 1}}}}
        nlp = English.from_config(config)
        nlp.add_pipe('dummy_component')
        nlp.initialize()