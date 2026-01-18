import pytest
import numpy as np
from cirq.contrib.qasm_import import QasmException
from cirq.contrib.qasm_import._lexer import QasmLexer
def test_creg():
    lexer = QasmLexer()
    lexer.input('creg [8];')
    token = lexer.token()
    assert token.type == 'CREG'
    assert token.value == 'creg'
    token = lexer.token()
    assert token.type == '['
    assert token.value == '['
    token = lexer.token()
    assert token.type == 'NATURAL_NUMBER'
    assert token.value == 8
    token = lexer.token()
    assert token.type == ']'
    assert token.value == ']'
    token = lexer.token()
    assert token.type == ';'
    assert token.value == ';'