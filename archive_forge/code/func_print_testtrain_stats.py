from collections import Counter, defaultdict
from nltk import jsontags
from nltk.tag import TaggerI
from nltk.tbl import Feature, Template
def print_testtrain_stats():
    testscores = test_stats['rulescores']
    print('TEMPLATE STATISTICS (TEST AND TRAIN) ({} templates, {} rules)'.format(len(template_counts), len(tids)))
    print('TEST  ({tokencount:7d} tokens) initial {initialerrors:5d} {initialacc:.4f} final: {finalerrors:5d} {finalacc:.4f} '.format(**test_stats))
    print('TRAIN ({tokencount:7d} tokens) initial {initialerrors:5d} {initialacc:.4f} final: {finalerrors:5d} {finalacc:.4f} '.format(**train_stats))
    weighted_testcounts = Counter()
    for tid, score in zip(tids, testscores):
        weighted_testcounts[tid] += score
    tottestscores = sum(testscores)
    head = '#ID | Score (test) | Score (train) |  #Rules     | Template'
    print(head, '\n', '-' * len(head), sep='')
    test_tplscores = sorted(weighted_testcounts.items(), key=det_tplsort, reverse=True)
    for tid, testscore in test_tplscores:
        s = '{:s} |{:5d}  {:6.3f} |  {:4d}   {:.3f} |{:4d}   {:.3f} | {:s}'.format(tid, testscore, testscore / tottestscores, weighted_traincounts[tid], weighted_traincounts[tid] / tottrainscores, template_counts[tid], template_counts[tid] / len(tids), Template.ALLTEMPLATES[int(tid)])
        print(s)