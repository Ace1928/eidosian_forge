from collections import Counter, defaultdict
from nltk import jsontags
from nltk.tag import TaggerI
from nltk.tbl import Feature, Template
def print_template_statistics(self, test_stats=None, printunused=True):
    """
        Print a list of all templates, ranked according to efficiency.

        If test_stats is available, the templates are ranked according to their
        relative contribution (summed for all rules created from a given template,
        weighted by score) to the performance on the test set. If no test_stats, then
        statistics collected during training are used instead. There is also
        an unweighted measure (just counting the rules). This is less informative,
        though, as many low-score rules will appear towards end of training.

        :param test_stats: dictionary of statistics collected during testing
        :type test_stats: dict of str -> any (but usually numbers)
        :param printunused: if True, print a list of all unused templates
        :type printunused: bool
        :return: None
        :rtype: None
        """
    tids = [r.templateid for r in self._rules]
    train_stats = self.train_stats()
    trainscores = train_stats['rulescores']
    assert len(trainscores) == len(tids), 'corrupt statistics: {} train scores for {} rules'.format(trainscores, tids)
    template_counts = Counter(tids)
    weighted_traincounts = Counter()
    for tid, score in zip(tids, trainscores):
        weighted_traincounts[tid] += score
    tottrainscores = sum(trainscores)

    def det_tplsort(tpl_value):
        return (tpl_value[1], repr(tpl_value[0]))

    def print_train_stats():
        print('TEMPLATE STATISTICS (TRAIN)  {} templates, {} rules)'.format(len(template_counts), len(tids)))
        print('TRAIN ({tokencount:7d} tokens) initial {initialerrors:5d} {initialacc:.4f} final: {finalerrors:5d} {finalacc:.4f}'.format(**train_stats))
        head = '#ID | Score (train) |  #Rules     | Template'
        print(head, '\n', '-' * len(head), sep='')
        train_tplscores = sorted(weighted_traincounts.items(), key=det_tplsort, reverse=True)
        for tid, trainscore in train_tplscores:
            s = '{} | {:5d}   {:5.3f} |{:4d}   {:.3f} | {}'.format(tid, trainscore, trainscore / tottrainscores, template_counts[tid], template_counts[tid] / len(tids), Template.ALLTEMPLATES[int(tid)])
            print(s)

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

    def print_unused_templates():
        usedtpls = {int(tid) for tid in tids}
        unused = [(tid, tpl) for tid, tpl in enumerate(Template.ALLTEMPLATES) if tid not in usedtpls]
        print(f'UNUSED TEMPLATES ({len(unused)})')
        for tid, tpl in unused:
            print(f'{tid:03d} {str(tpl):s}')
    if test_stats is None:
        print_train_stats()
    else:
        print_testtrain_stats()
    print()
    if printunused:
        print_unused_templates()
    print()