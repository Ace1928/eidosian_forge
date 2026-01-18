import re
from IPython.core.hooks import CommandChainDispatcher
def flat_matches(self, key):
    """ Yield all 'value' targets, without priority """
    for val in self.dispatch(key):
        for el in val:
            yield el[1]
    return