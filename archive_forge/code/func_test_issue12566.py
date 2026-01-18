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
@pytest.mark.issue(12566)
@pytest.mark.parametrize('factory,output_file', [('deps', 'parses.html'), ('ents', 'entities.html'), ('spans', 'spans.html')])
def test_issue12566(factory: str, output_file: str):
    """
    Test if all displaCy types (ents, dep, spans) produce an HTML file
    """
    with make_tempdir() as tmp_dir:
        doc_json = {'ents': [{'end': 54, 'label': 'nam_adj_country', 'start': 44}, {'end': 83, 'label': 'nam_liv_person', 'start': 69}, {'end': 100, 'label': 'nam_pro_title_book', 'start': 86}], 'spans': {'sc': [{'end': 54, 'kb_id': '', 'label': 'nam_adj_country', 'start': 44}, {'end': 83, 'kb_id': '', 'label': 'nam_liv_person', 'start': 69}, {'end': 100, 'kb_id': '', 'label': 'nam_pro_title_book', 'start': 86}]}, 'text': 'Niedawno czytał em nową książkę znakomitego szkockiego medioznawcy , Briana McNaira - Cultural Chaos .', 'tokens': [{'id': 0, 'start': 0, 'end': 8, 'tag': 'ADV', 'pos': 'ADV', 'morph': 'Degree=Pos', 'lemma': 'niedawno', 'dep': 'advmod', 'head': 1}, {'id': 1, 'start': 9, 'end': 15, 'tag': 'PRAET', 'pos': 'VERB', 'morph': 'Animacy=Hum|Aspect=Imp|Gender=Masc|Mood=Ind|Number=Sing|Tense=Past|VerbForm=Fin|Voice=Act', 'lemma': 'czytać', 'dep': 'ROOT', 'head': 1}, {'id': 2, 'start': 16, 'end': 18, 'tag': 'AGLT', 'pos': 'NOUN', 'morph': 'Animacy=Inan|Case=Ins|Gender=Masc|Number=Sing', 'lemma': 'em', 'dep': 'iobj', 'head': 1}, {'id': 3, 'start': 19, 'end': 23, 'tag': 'ADJ', 'pos': 'ADJ', 'morph': 'Case=Acc|Degree=Pos|Gender=Fem|Number=Sing', 'lemma': 'nowy', 'dep': 'amod', 'head': 4}, {'id': 4, 'start': 24, 'end': 31, 'tag': 'SUBST', 'pos': 'NOUN', 'morph': 'Case=Acc|Gender=Fem|Number=Sing', 'lemma': 'książka', 'dep': 'obj', 'head': 1}, {'id': 5, 'start': 32, 'end': 43, 'tag': 'ADJ', 'pos': 'ADJ', 'morph': 'Animacy=Nhum|Case=Gen|Degree=Pos|Gender=Masc|Number=Sing', 'lemma': 'znakomit', 'dep': 'acl', 'head': 4}, {'id': 6, 'start': 44, 'end': 54, 'tag': 'ADJ', 'pos': 'ADJ', 'morph': 'Animacy=Hum|Case=Gen|Degree=Pos|Gender=Masc|Number=Sing', 'lemma': 'szkockiy', 'dep': 'amod', 'head': 7}, {'id': 7, 'start': 55, 'end': 66, 'tag': 'SUBST', 'pos': 'NOUN', 'morph': 'Animacy=Hum|Case=Gen|Gender=Masc|Number=Sing', 'lemma': 'medioznawca', 'dep': 'iobj', 'head': 5}, {'id': 8, 'start': 67, 'end': 68, 'tag': 'INTERP', 'pos': 'PUNCT', 'morph': 'PunctType=Comm', 'lemma': ',', 'dep': 'punct', 'head': 9}, {'id': 9, 'start': 69, 'end': 75, 'tag': 'SUBST', 'pos': 'PROPN', 'morph': 'Animacy=Hum|Case=Gen|Gender=Masc|Number=Sing', 'lemma': 'Brian', 'dep': 'nmod', 'head': 4}, {'id': 10, 'start': 76, 'end': 83, 'tag': 'SUBST', 'pos': 'PROPN', 'morph': 'Animacy=Hum|Case=Gen|Gender=Masc|Number=Sing', 'lemma': 'McNair', 'dep': 'flat', 'head': 9}, {'id': 11, 'start': 84, 'end': 85, 'tag': 'INTERP', 'pos': 'PUNCT', 'morph': 'PunctType=Dash', 'lemma': '-', 'dep': 'punct', 'head': 12}, {'id': 12, 'start': 86, 'end': 94, 'tag': 'SUBST', 'pos': 'PROPN', 'morph': 'Animacy=Inan|Case=Nom|Gender=Masc|Number=Sing', 'lemma': 'Cultural', 'dep': 'conj', 'head': 4}, {'id': 13, 'start': 95, 'end': 100, 'tag': 'SUBST', 'pos': 'NOUN', 'morph': 'Animacy=Inan|Case=Nom|Gender=Masc|Number=Sing', 'lemma': 'Chaos', 'dep': 'flat', 'head': 12}, {'id': 14, 'start': 101, 'end': 102, 'tag': 'INTERP', 'pos': 'PUNCT', 'morph': 'PunctType=Peri', 'lemma': '.', 'dep': 'punct', 'head': 1}]}
        nlp = spacy.blank('pl')
        doc = Doc(nlp.vocab).from_json(doc_json)
        render_parses(docs=[doc], output_path=tmp_dir, model_name='', limit=1, **{factory: True})
        assert (tmp_dir / output_file).is_file()