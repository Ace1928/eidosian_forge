from math import log
from operator import itemgetter
from nltk.probability import ConditionalFreqDist, FreqDist
from nltk.tag.api import TaggerI
def demo3():
    from nltk.corpus import brown, treebank
    d = list(treebank.tagged_sents())
    e = list(brown.tagged_sents())
    d = d[:1000]
    e = e[:1000]
    d10 = int(len(d) * 0.1)
    e10 = int(len(e) * 0.1)
    tknacc = 0
    sknacc = 0
    tallacc = 0
    sallacc = 0
    tknown = 0
    sknown = 0
    for i in range(10):
        t = TnT(N=1000, C=False)
        s = TnT(N=1000, C=False)
        dtest = d[i * d10:(i + 1) * d10]
        etest = e[i * e10:(i + 1) * e10]
        dtrain = d[:i * d10] + d[(i + 1) * d10:]
        etrain = e[:i * e10] + e[(i + 1) * e10:]
        t.train(dtrain)
        s.train(etrain)
        tacc = t.accuracy(dtest)
        tp_un = t.unknown / (t.known + t.unknown)
        tp_kn = t.known / (t.known + t.unknown)
        tknown += tp_kn
        t.unknown = 0
        t.known = 0
        sacc = s.accuracy(etest)
        sp_un = s.unknown / (s.known + s.unknown)
        sp_kn = s.known / (s.known + s.unknown)
        sknown += sp_kn
        s.unknown = 0
        s.known = 0
        tknacc += tacc / tp_kn
        sknacc += sacc / tp_kn
        tallacc += tacc
        sallacc += sacc
    print('brown: acc over words known:', 10 * tknacc)
    print('     : overall accuracy:', 10 * tallacc)
    print('     : words known:', 10 * tknown)
    print('treebank: acc over words known:', 10 * sknacc)
    print('        : overall accuracy:', 10 * sallacc)
    print('        : words known:', 10 * sknown)