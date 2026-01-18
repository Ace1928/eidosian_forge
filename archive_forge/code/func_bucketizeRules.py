from fontTools import ttLib
from fontTools.ttLib.tables._c_m_a_p import cmap_classes
from fontTools.ttLib.tables import otTables as ot
from fontTools.ttLib.tables.otBase import ValueRecord, valueRecordFormatDict
from fontTools.otlLib import builder as otl
from contextlib import contextmanager
from fontTools.ttLib import newTable
from fontTools.feaLib.lookupDebugInfo import LOOKUP_DEBUG_ENV_VAR, LOOKUP_DEBUG_INFO_KEY
from operator import setitem
import os
import logging
def bucketizeRules(self, c, rules, bucketKeys):
    buckets = {}
    for seq, recs in rules:
        buckets.setdefault(seq[c.InputIdx][0], []).append((tuple((s[1 if i == c.InputIdx else 0:] for i, s in enumerate(seq))), recs))
    rulesets = []
    for firstGlyph in bucketKeys:
        if firstGlyph not in buckets:
            rulesets.append(None)
            continue
        thisRules = []
        for seq, recs in buckets[firstGlyph]:
            rule = getattr(ot, c.Rule)()
            c.SetRuleData(rule, seq)
            setattr(rule, c.Type + 'Count', len(recs))
            setattr(rule, c.LookupRecord, recs)
            thisRules.append(rule)
        ruleset = getattr(ot, c.RuleSet)()
        setattr(ruleset, c.Rule, thisRules)
        setattr(ruleset, c.RuleCount, len(thisRules))
        rulesets.append(ruleset)
    setattr(self, c.RuleSet, rulesets)
    setattr(self, c.RuleSetCount, len(rulesets))