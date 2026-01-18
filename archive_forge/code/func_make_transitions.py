import sys
import re
import types
import unicodedata
from docutils import utils
from docutils.utils.error_reporting import ErrorOutput
def make_transitions(self, name_list):
    """
        Return a list of transition names and a transition mapping.

        Parameter `name_list`: a list, where each entry is either a transition
        name string, or a 1- or 2-tuple (transition name, optional next state
        name).
        """
    stringtype = type('')
    names = []
    transitions = {}
    for namestate in name_list:
        if type(namestate) is stringtype:
            transitions[namestate] = self.make_transition(namestate)
            names.append(namestate)
        else:
            transitions[namestate[0]] = self.make_transition(*namestate)
            names.append(namestate[0])
    return (names, transitions)