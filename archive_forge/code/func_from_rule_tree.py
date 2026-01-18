import os
import re
import stat
import sys
import fnmatch
from xdg import BaseDirectory
import xdg.Locale
from xml.dom import minidom, XML_NAMESPACE
from collections import defaultdict
@classmethod
def from_rule_tree(cls, tree):
    """From a nested list of (rule, subrules) pairs, build a MagicMatchAny
        instance, recursing down the tree.
        
        Where there's only one top-level rule, this is returned directly,
        to simplify the nested structure. Returns None if no rules were read.
        """
    rules = []
    for rule, subrules in tree:
        if subrules:
            rule.also = cls.from_rule_tree(subrules)
        rules.append(rule)
    if len(rules) == 0:
        return None
    if len(rules) == 1:
        return rules[0]
    return cls(rules)