from other tagging techniques which often tag each word individually, seeking
import itertools
import re
from nltk.metrics import accuracy
from nltk.probability import (
from nltk.tag.api import TaggerI
from nltk.util import LazyMap, unique_list
def demo_bw():
    print()
    print('Baum-Welch demo for market example')
    print()
    model, states, symbols = _market_hmm_example()
    training = []
    import random
    rng = random.Random()
    rng.seed(0)
    for i in range(10):
        item = model.random_sample(rng, 5)
        training.append([(i[0], None) for i in item])
    trainer = HiddenMarkovModelTrainer(states, symbols)
    hmm = trainer.train_unsupervised(training, model=model, max_iterations=1000)