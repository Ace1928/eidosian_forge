import inspect
import itertools
import logging
import warnings
from collections import OrderedDict, defaultdict, deque
from functools import partial
from six import string_types
def remove_transition(self, trigger, source='*', dest='*'):
    """ Removes a transition from the Machine and all models.
        Args:
            trigger (str): Trigger name of the transition.
            source (str, Enum or State): Limits removal to transitions from a certain state.
            dest (str, Enum or State): Limits removal to transitions to a certain state.
        """
    source = listify(source) if source != '*' else source
    dest = listify(dest) if dest != '*' else dest
    tmp = {key: value for key, value in {k: [t for t in v if source != '*' and t.source not in source or (dest != '*' and t.dest not in dest)] for k, v in self.events[trigger].transitions.items()}.items() if len(value) > 0}
    if tmp:
        self.events[trigger].transitions = defaultdict(list, **tmp)
    else:
        for model in self.models:
            delattr(model, trigger)
        del self.events[trigger]