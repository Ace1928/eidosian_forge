import dataclasses
import itertools
from typing import (
import networkx as nx
import numpy as np
from cirq import circuits, devices, ops, protocols, value
from cirq._doc import document
def pair_gen():
    for moment in layer_circuit.moments:
        yield (_pairs_from_moment(moment), moment)