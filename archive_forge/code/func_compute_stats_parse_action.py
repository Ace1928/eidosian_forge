from __future__ import division
import pyparsing as pp
from collections import namedtuple
from datetime import datetime
def compute_stats_parse_action(t):
    t['sum'] = sum(t)
    t['ave'] = sum(t) / len(t)
    t['min'] = min(t)
    t['max'] = max(t)