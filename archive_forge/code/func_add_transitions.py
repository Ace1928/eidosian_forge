import inspect
import itertools
import logging
import warnings
from collections import OrderedDict, defaultdict, deque
from functools import partial
from six import string_types
def add_transitions(self, transitions):
    """ Add several transitions.

        Args:
            transitions (list): A list of transitions.

        """
    for trans in listify(transitions):
        if isinstance(trans, list):
            self.add_transition(*trans)
        else:
            self.add_transition(**trans)