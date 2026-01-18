import re
from typing import Dict, Any
def compile_sc(self, rules=None):
    if rules is None:
        key = '$'
        rules = self.rules
    else:
        key = '|'.join(rules)
    sc = self.__sc.get(key)
    if sc:
        return sc
    regex = '|'.join(('(?P<%s>%s)' % (k, self.specification[k]) for k in rules))
    sc = re.compile(regex, self.sc_flag)
    self.__sc[key] = sc
    return sc