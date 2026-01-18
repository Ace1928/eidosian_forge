from collections import deque
from sympy.combinatorics.rewritingsystem_fsm import StateMachine
def make_confluent(self, check=False):
    """
        Try to make the system confluent using the Knuth-Bendix
        completion algorithm

        """
    if self._max_exceeded:
        return self._is_confluent
    lhs = list(self.rules.keys())

    def _overlaps(r1, r2):
        len1 = len(r1)
        len2 = len(r2)
        result = []
        for j in range(1, len1 + len2):
            if r1.subword(len1 - j, len1 + len2 - j, strict=False) == r2.subword(j - len1, j, strict=False):
                a = r1.subword(0, len1 - j, strict=False)
                a = a * r2.subword(0, j - len1, strict=False)
                b = r2.subword(j - len1, j, strict=False)
                c = r2.subword(j, len2, strict=False)
                c = c * r1.subword(len1 + len2 - j, len1, strict=False)
                result.append(a * b * c)
        return result

    def _process_overlap(w, r1, r2, check):
        s = w.eliminate_word(r1, self.rules[r1])
        s = self.reduce(s)
        t = w.eliminate_word(r2, self.rules[r2])
        t = self.reduce(t)
        if s != t:
            if check:
                return [0]
            try:
                new_keys = self.add_rule(t, s, check)
                return new_keys
            except RuntimeError:
                return False
        return
    added = 0
    i = 0
    while i < len(lhs):
        r1 = lhs[i]
        i += 1
        j = 0
        while j < len(lhs):
            r2 = lhs[j]
            j += 1
            if r1 == r2:
                continue
            overlaps = _overlaps(r1, r2)
            overlaps.extend(_overlaps(r1 ** (-1), r2))
            if not overlaps:
                continue
            for w in overlaps:
                new_keys = _process_overlap(w, r1, r2, check)
                if new_keys:
                    if check:
                        return False
                    lhs.extend(new_keys)
                    added += len(new_keys)
                elif new_keys == False:
                    return self._is_confluent
            if added > self.tidyint and (not check):
                r, a = self._remove_redundancies(changes=True)
                added = 0
                if r:
                    i = min([lhs.index(s) for s in r])
                lhs = [l for l in lhs if l not in r]
                lhs.extend(a)
                if r1 in r:
                    break
    self._is_confluent = True
    if not check:
        self._remove_redundancies()
    return True