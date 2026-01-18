import cirq
import cirq_web
import pytest
def strip_ws(string):
    return ''.join(string.split())