import itertools
import cirq
import cirq_ft
from cirq_ft import infra
import numpy as np
import pytest
from cirq_ft.deprecation import allow_deprecated_cirq_ft_use_in_tests
def get_3q_uniform_dirac_notation(signs):
    terms = ['0.35|000⟩', '0.35|001⟩', '0.35|010⟩', '0.35|011⟩', '0.35|100⟩', '0.35|101⟩', '0.35|110⟩', '0.35|111⟩']
    ret = terms[0] if signs[0] == '+' else f'-{terms[0]}'
    for c, term in zip(signs[1:], terms[1:]):
        ret = ret + f' {c} {term}'
    return ret