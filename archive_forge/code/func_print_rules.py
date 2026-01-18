from collections import defaultdict
from typing import Iterator
from .logic import Logic, And, Or, Not
def print_rules(self) -> Iterator[str]:
    """ Returns a generator with lines to represent the facts and rules """
    yield from self._defined_facts_lines()
    yield ''
    yield ''
    yield from self._full_implications_lines()
    yield ''
    yield ''
    yield from self._prereq_lines()
    yield ''
    yield ''
    yield from self._beta_rules_lines()
    yield ''
    yield ''
    yield "generated_assumptions = {'defined_facts': defined_facts, 'full_implications': full_implications,"
    yield "               'prereq': prereq, 'beta_rules': beta_rules, 'beta_triggers': beta_triggers}"