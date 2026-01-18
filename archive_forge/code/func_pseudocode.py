from collections import defaultdict
from nltk.classify.api import ClassifierI
from nltk.probability import FreqDist, MLEProbDist, entropy
def pseudocode(self, prefix='', depth=4):
    """
        Return a string representation of this decision tree that
        expresses the decisions it makes as a nested set of pseudocode
        if statements.
        """
    if self._fname is None:
        return f'{prefix}return {self._label!r}\n'
    s = ''
    for fval, result in sorted(self._decisions.items(), key=lambda item: (item[0] in [None, False, True], str(item[0]).lower())):
        s += f'{prefix}if {self._fname} == {fval!r}: '
        if result._fname is not None and depth > 1:
            s += '\n' + result.pseudocode(prefix + '  ', depth - 1)
        else:
            s += f'return {result._label!r}\n'
    if self._default is not None:
        if len(self._decisions) == 1:
            s += '{}if {} != {!r}: '.format(prefix, self._fname, list(self._decisions.keys())[0])
        else:
            s += f'{prefix}else: '
        if self._default._fname is not None and depth > 1:
            s += '\n' + self._default.pseudocode(prefix + '  ', depth - 1)
        else:
            s += f'return {self._default._label!r}\n'
    return s