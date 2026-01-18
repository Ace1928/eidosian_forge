import math
import os
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple
import pytest
import srsly
from click import NoSuchOption
from packaging.specifiers import SpecifierSet
from thinc.api import Config
import spacy
from spacy import about
from spacy.cli import info
from spacy.cli._util import parse_config_overrides, string_to_list, walk_directory
from spacy.cli.apply import apply
from spacy.cli.debug_data import (
from spacy.cli.download import get_compatibility, get_version
from spacy.cli.evaluate import render_parses
from spacy.cli.find_threshold import find_threshold
from spacy.cli.init_config import RECOMMENDATIONS, fill_config, init_config
from spacy.cli.init_pipeline import _init_labels
from spacy.cli.package import _is_permitted_package_name, get_third_party_dependencies
from spacy.cli.validate import get_model_pkgs
from spacy.lang.en import English
from spacy.lang.nl import Dutch
from spacy.language import Language
from spacy.schemas import RecommendationSchema
from spacy.tokens import Doc, DocBin
from spacy.tokens.span import Span
from spacy.training import Example, docs_to_json, offsets_to_biluo_tags
from spacy.training.converters import conll_ner_to_docs, conllu_to_docs, iob_to_docs
from spacy.util import ENV_VARS, get_minor_version, load_config, load_model_from_config
from .util import make_tempdir
@pytest.mark.parametrize('lang', ['en', 'nl'])
@pytest.mark.parametrize('pipeline', [['tagger', 'parser', 'ner'], [], ['ner', 'textcat', 'sentencizer'], ['morphologizer', 'spancat', 'entity_linker'], ['spancat_singlelabel', 'textcat_multilabel']])
@pytest.mark.parametrize('optimize', ['efficiency', 'accuracy'])
@pytest.mark.parametrize('pretraining', [True, False])
def test_init_config(lang, pipeline, optimize, pretraining):
    config = init_config(lang=lang, pipeline=pipeline, optimize=optimize, pretraining=pretraining, gpu=False)
    assert isinstance(config, Config)
    if pretraining:
        config['paths']['raw_text'] = 'my_data.jsonl'
    load_model_from_config(config, auto_fill=True)