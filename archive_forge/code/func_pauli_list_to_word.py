import warnings
from collections.abc import Iterable
from string import ascii_letters as ABC
import numpy as np
import pennylane as qml
def pauli_list_to_word(obs):
    word = [-1] * num_wires
    for ob in obs:
        if ob.name not in obs_to_recipe_map:
            raise ValueError('Observable must be a linear combination of Pauli observables')
        word[self.wire_map.index(ob.wires[0])] = obs_to_recipe_map[ob.name]
    return word