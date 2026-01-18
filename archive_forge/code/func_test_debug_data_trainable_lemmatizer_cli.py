import os
import sys
from pathlib import Path
import pytest
import srsly
from typer.testing import CliRunner
from spacy.cli._util import app, get_git_version
from spacy.tokens import Doc, DocBin, Span
from .util import make_tempdir, normalize_whitespace
def test_debug_data_trainable_lemmatizer_cli(en_vocab):
    train_docs = [Doc(en_vocab, words=['I', 'like', 'cats'], lemmas=['I', 'like', 'cat']), Doc(en_vocab, words=['Dogs', 'are', 'great', 'too'], lemmas=['dog', 'be', 'great', 'too'])]
    dev_docs = [Doc(en_vocab, words=['Cats', 'are', 'cute'], lemmas=['cat', 'be', 'cute']), Doc(en_vocab, words=['Pets', 'are', 'great'], lemmas=['pet', 'be', 'great'])]
    with make_tempdir() as d_in:
        train_bin = DocBin(docs=train_docs)
        train_bin.to_disk(d_in / 'train.spacy')
        dev_bin = DocBin(docs=dev_docs)
        dev_bin.to_disk(d_in / 'dev.spacy')
        CliRunner().invoke(app, ['init', 'config', f'{d_in}/config.cfg', '--lang', 'en', '--pipeline', 'trainable_lemmatizer'])
        result_debug_data = CliRunner().invoke(app, ['debug', 'data', f'{d_in}/config.cfg', '--paths.train', f'{d_in}/train.spacy', '--paths.dev', f'{d_in}/dev.spacy'])
        assert '= Trainable Lemmatizer =' in result_debug_data.stdout