from functools import reduce
from pyparsing import Word, OneOrMore, Optional, Literal, NotAny, MatchFirst
from pyparsing import Group, oneOf, Suppress, ZeroOrMore, Combine, FollowedBy
from pyparsing import srange, CharsNotIn, StringEnd, LineEnd, White, Regex
from pyparsing import nums, alphas, alphanums, ParseException, Forward
import types, sys, os, re, datetime
def getXmlDocs(abc_string, skip=0, num=1, rOpt=False, bOpt=False, fOpt=False):
    xml_docs = []
    abctext = expand_abc_include(abc_string)
    fragments = re.split('^\\s*X:', abctext, flags=re.M)
    preamble = fragments[0]
    tunes = fragments[1:]
    if not tunes and preamble:
        tunes, preamble = (['1\n' + preamble], '')
    for itune, tune in enumerate(tunes):
        if itune < skip:
            continue
        if itune >= skip + num:
            break
        tune = preamble + 'X:' + tune
        try:
            score = mxm.parse(tune, rOpt, bOpt, fOpt)
            ds = list(score.iter('duration'))
            ss = [int(d.text) for d in ds]
            deler = reduce(ggd, ss + [21])
            for i, d in enumerate(ds):
                d.text = str(ss[i] // deler)
            for d in score.iter('divisions'):
                d.text = str(int(d.text) // deler)
            xml_docs.append(score)
        except ParseException:
            pass
        except Exception as err:
            info('an exception occurred.\n%s' % err)
    return xml_docs