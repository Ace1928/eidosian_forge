import os
import sys
from pathlib import Path
import pytest
import srsly
from typer.testing import CliRunner
from spacy.cli._util import app, get_git_version
from spacy.tokens import Doc, DocBin, Span
from .util import make_tempdir, normalize_whitespace
@pytest.mark.slow
@pytest.mark.parametrize('component,examples', [('tagger', [TRAIN_EXAMPLE_1, TRAIN_EXAMPLE_2]), ('morphologizer', [TRAIN_EXAMPLE_1, TRAIN_EXAMPLE_2]), ('trainable_lemmatizer', [TRAIN_EXAMPLE_1, TRAIN_EXAMPLE_2]), ('parser', [TRAIN_EXAMPLE_1] * 30), ('ner', [TRAIN_EXAMPLE_1, TRAIN_EXAMPLE_2]), ('spancat', [TRAIN_EXAMPLE_1, TRAIN_EXAMPLE_2]), ('textcat', [TRAIN_EXAMPLE_1, TRAIN_EXAMPLE_2])])
def test_init_config_trainable(component, examples, en_vocab):
    if component == 'textcat':
        train_docs = []
        for example in examples:
            doc = Doc(en_vocab, words=example['words'])
            doc.cats = example['cats']
            train_docs.append(doc)
    elif component == 'spancat':
        train_docs = []
        for example in examples:
            doc = Doc(en_vocab, words=example['words'])
            doc.spans['sc'] = [Span(doc, start, end, label) for start, end, label in example['spans']]
            train_docs.append(doc)
    else:
        train_docs = []
        for example in examples:
            example = {k: v for k, v in example.items() if k not in ('cats', 'spans')}
            doc = Doc(en_vocab, **example)
            train_docs.append(doc)
    with make_tempdir() as d_in:
        train_bin = DocBin(docs=train_docs)
        train_bin.to_disk(d_in / 'train.spacy')
        dev_bin = DocBin(docs=train_docs)
        dev_bin.to_disk(d_in / 'dev.spacy')
        init_config_result = CliRunner().invoke(app, ['init', 'config', f'{d_in}/config.cfg', '--lang', 'en', '--pipeline', component])
        assert init_config_result.exit_code == 0
        train_result = CliRunner().invoke(app, ['train', f'{d_in}/config.cfg', '--paths.train', f'{d_in}/train.spacy', '--paths.dev', f'{d_in}/dev.spacy', '--output', f'{d_in}/model'])
        assert train_result.exit_code == 0
        assert Path(d_in / 'model' / 'model-last').exists()