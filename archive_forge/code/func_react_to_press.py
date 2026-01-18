import collections
import functools
import random
from automaton import exceptions as excp
from automaton import machines
from automaton import runners
from testtools import testcase
def react_to_press(last_state, new_state, event, number_calling):
    if len(number_calling) >= 10:
        return 'call'
    else:
        return 'press'