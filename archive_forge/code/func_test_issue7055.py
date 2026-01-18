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
@pytest.mark.issue(7055)
def test_issue7055():
    """Test that fill-config doesn't turn sourced components into factories."""
    source_cfg = {'nlp': {'lang': 'en', 'pipeline': ['tok2vec', 'tagger']}, 'components': {'tok2vec': {'factory': 'tok2vec'}, 'tagger': {'factory': 'tagger'}}}
    source_nlp = English.from_config(source_cfg)
    with make_tempdir() as dir_path:
        source_path = dir_path / 'test_model'
        source_nlp.to_disk(source_path)
        base_cfg = {'nlp': {'lang': 'en', 'pipeline': ['tok2vec', 'tagger', 'ner']}, 'components': {'tok2vec': {'source': str(source_path)}, 'tagger': {'source': str(source_path)}, 'ner': {'factory': 'ner'}}}
        base_cfg = Config(base_cfg)
        base_path = dir_path / 'base.cfg'
        base_cfg.to_disk(base_path)
        output_path = dir_path / 'config.cfg'
        fill_config(output_path, base_path, silent=True)
        filled_cfg = load_config(output_path)
    assert filled_cfg['components']['tok2vec']['source'] == str(source_path)
    assert filled_cfg['components']['tagger']['source'] == str(source_path)
    assert filled_cfg['components']['ner']['factory'] == 'ner'
    assert 'model' in filled_cfg['components']['ner']