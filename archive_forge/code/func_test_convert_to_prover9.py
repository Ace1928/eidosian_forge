import os
import subprocess
import nltk
from nltk.inference.api import BaseProverCommand, Prover
from nltk.sem.logic import (
def test_convert_to_prover9(expr):
    """
    Test that parsing works OK.
    """
    for t in expr:
        e = Expression.fromstring(t)
        print(convert_to_prover9(e))