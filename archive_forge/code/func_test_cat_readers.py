from typing import Callable, Dict, Iterable
import pytest
from thinc.api import Config, fix_random_seed
from spacy import Language
from spacy.schemas import ConfigSchemaTraining
from spacy.training import Example
from spacy.util import load_model_from_config, registry, resolve_dot_names
@pytest.mark.slow
@pytest.mark.parametrize('reader,additional_config', [('ml_datasets.imdb_sentiment.v1', {'train_limit': 10, 'dev_limit': 10}), ('ml_datasets.dbpedia.v1', {'train_limit': 10, 'dev_limit': 10}), ('ml_datasets.cmu_movies.v1', {'limit': 10, 'freq_cutoff': 200, 'split': 0.8})])
def test_cat_readers(reader, additional_config):
    nlp_config_string = '\n    [training]\n    seed = 0\n\n    [training.score_weights]\n    cats_macro_auc = 1.0\n\n    [corpora]\n    @readers = "PLACEHOLDER"\n\n    [nlp]\n    lang = "en"\n    pipeline = ["tok2vec", "textcat_multilabel"]\n\n    [components]\n\n    [components.tok2vec]\n    factory = "tok2vec"\n\n    [components.textcat_multilabel]\n    factory = "textcat_multilabel"\n    '
    config = Config().from_str(nlp_config_string)
    fix_random_seed(config['training']['seed'])
    config['corpora']['@readers'] = reader
    config['corpora'].update(additional_config)
    nlp = load_model_from_config(config, auto_fill=True)
    T = registry.resolve(nlp.config['training'], schema=ConfigSchemaTraining)
    dot_names = [T['train_corpus'], T['dev_corpus']]
    train_corpus, dev_corpus = resolve_dot_names(nlp.config, dot_names)
    optimizer = T['optimizer']
    nlp.initialize(lambda: train_corpus(nlp), sgd=optimizer)
    for example in train_corpus(nlp):
        assert example.y.cats
        assert sorted(list(set(example.y.cats.values()))) == [0.0, 1.0]
        nlp.update([example], sgd=optimizer)
    dev_examples = list(dev_corpus(nlp))
    for example in dev_examples:
        assert sorted(list(set(example.y.cats.values()))) == [0.0, 1.0]
    scores = nlp.evaluate(dev_examples)
    assert scores['cats_score']
    doc = nlp('Quick test')
    assert doc.cats