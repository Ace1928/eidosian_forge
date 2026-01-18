import os
import pickle
import random
import time
from nltk.corpus import treebank
from nltk.tag import BrillTaggerTrainer, RegexpTagger, UnigramTagger
from nltk.tag.brill import Pos, Word
from nltk.tbl import Template, error_list
def postag(templates=None, tagged_data=None, num_sents=1000, max_rules=300, min_score=3, min_acc=None, train=0.8, trace=3, randomize=False, ruleformat='str', incremental_stats=False, template_stats=False, error_output=None, serialize_output=None, learning_curve_output=None, learning_curve_take=300, baseline_backoff_tagger=None, separate_baseline_data=False, cache_baseline_tagger=None):
    """
    Brill Tagger Demonstration
    :param templates: how many sentences of training and testing data to use
    :type templates: list of Template

    :param tagged_data: maximum number of rule instances to create
    :type tagged_data: C{int}

    :param num_sents: how many sentences of training and testing data to use
    :type num_sents: C{int}

    :param max_rules: maximum number of rule instances to create
    :type max_rules: C{int}

    :param min_score: the minimum score for a rule in order for it to be considered
    :type min_score: C{int}

    :param min_acc: the minimum score for a rule in order for it to be considered
    :type min_acc: C{float}

    :param train: the fraction of the the corpus to be used for training (1=all)
    :type train: C{float}

    :param trace: the level of diagnostic tracing output to produce (0-4)
    :type trace: C{int}

    :param randomize: whether the training data should be a random subset of the corpus
    :type randomize: C{bool}

    :param ruleformat: rule output format, one of "str", "repr", "verbose"
    :type ruleformat: C{str}

    :param incremental_stats: if true, will tag incrementally and collect stats for each rule (rather slow)
    :type incremental_stats: C{bool}

    :param template_stats: if true, will print per-template statistics collected in training and (optionally) testing
    :type template_stats: C{bool}

    :param error_output: the file where errors will be saved
    :type error_output: C{string}

    :param serialize_output: the file where the learned tbl tagger will be saved
    :type serialize_output: C{string}

    :param learning_curve_output: filename of plot of learning curve(s) (train and also test, if available)
    :type learning_curve_output: C{string}

    :param learning_curve_take: how many rules plotted
    :type learning_curve_take: C{int}

    :param baseline_backoff_tagger: the file where rules will be saved
    :type baseline_backoff_tagger: tagger

    :param separate_baseline_data: use a fraction of the training data exclusively for training baseline
    :type separate_baseline_data: C{bool}

    :param cache_baseline_tagger: cache baseline tagger to this file (only interesting as a temporary workaround to get
                                  deterministic output from the baseline unigram tagger between python versions)
    :type cache_baseline_tagger: C{string}


    Note on separate_baseline_data: if True, reuse training data both for baseline and rule learner. This
    is fast and fine for a demo, but is likely to generalize worse on unseen data.
    Also cannot be sensibly used for learning curves on training data (the baseline will be artificially high).
    """
    baseline_backoff_tagger = baseline_backoff_tagger or REGEXP_TAGGER
    if templates is None:
        from nltk.tag.brill import brill24, describe_template_sets
        templates = brill24()
    training_data, baseline_data, gold_data, testing_data = _demo_prepare_data(tagged_data, train, num_sents, randomize, separate_baseline_data)
    if cache_baseline_tagger:
        if not os.path.exists(cache_baseline_tagger):
            baseline_tagger = UnigramTagger(baseline_data, backoff=baseline_backoff_tagger)
            with open(cache_baseline_tagger, 'w') as print_rules:
                pickle.dump(baseline_tagger, print_rules)
            print('Trained baseline tagger, pickled it to {}'.format(cache_baseline_tagger))
        with open(cache_baseline_tagger) as print_rules:
            baseline_tagger = pickle.load(print_rules)
            print(f'Reloaded pickled tagger from {cache_baseline_tagger}')
    else:
        baseline_tagger = UnigramTagger(baseline_data, backoff=baseline_backoff_tagger)
        print('Trained baseline tagger')
    if gold_data:
        print('    Accuracy on test set: {:0.4f}'.format(baseline_tagger.accuracy(gold_data)))
    tbrill = time.time()
    trainer = BrillTaggerTrainer(baseline_tagger, templates, trace, ruleformat=ruleformat)
    print('Training tbl tagger...')
    brill_tagger = trainer.train(training_data, max_rules, min_score, min_acc)
    print(f'Trained tbl tagger in {time.time() - tbrill:0.2f} seconds')
    if gold_data:
        print('    Accuracy on test set: %.4f' % brill_tagger.accuracy(gold_data))
    if trace == 1:
        print('\nLearned rules: ')
        for ruleno, rule in enumerate(brill_tagger.rules(), 1):
            print(f'{ruleno:4d} {rule.format(ruleformat):s}')
    if incremental_stats:
        print('Incrementally tagging the test data, collecting individual rule statistics')
        taggedtest, teststats = brill_tagger.batch_tag_incremental(testing_data, gold_data)
        print('    Rule statistics collected')
        if not separate_baseline_data:
            print('WARNING: train_stats asked for separate_baseline_data=True; the baseline will be artificially high')
        trainstats = brill_tagger.train_stats()
        if template_stats:
            brill_tagger.print_template_statistics(teststats)
        if learning_curve_output:
            _demo_plot(learning_curve_output, teststats, trainstats, take=learning_curve_take)
            print(f'Wrote plot of learning curve to {learning_curve_output}')
    else:
        print('Tagging the test data')
        taggedtest = brill_tagger.tag_sents(testing_data)
        if template_stats:
            brill_tagger.print_template_statistics()
    if error_output is not None:
        with open(error_output, 'w') as f:
            f.write('Errors for Brill Tagger %r\n\n' % serialize_output)
            f.write('\n'.join(error_list(gold_data, taggedtest)).encode('utf-8') + '\n')
        print(f'Wrote tagger errors including context to {error_output}')
    if serialize_output is not None:
        taggedtest = brill_tagger.tag_sents(testing_data)
        with open(serialize_output, 'w') as print_rules:
            pickle.dump(brill_tagger, print_rules)
        print(f'Wrote pickled tagger to {serialize_output}')
        with open(serialize_output) as print_rules:
            brill_tagger_reloaded = pickle.load(print_rules)
        print(f'Reloaded pickled tagger from {serialize_output}')
        taggedtest_reloaded = brill_tagger.tag_sents(testing_data)
        if taggedtest == taggedtest_reloaded:
            print('Reloaded tagger tried on test set, results identical')
        else:
            print('PROBLEM: Reloaded tagger gave different results on test set')