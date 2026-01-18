from __future__ import absolute_import, division, print_function
import copy
def facts(self):
    facts_result = dict(((k, v) for k, v in self._data.items() if self._meta[k].fact))
    return facts_result if facts_result else None