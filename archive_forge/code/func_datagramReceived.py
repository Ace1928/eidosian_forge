from __future__ import annotations
from zope import interface
from twisted.pair import ip, raw
from twisted.python import components
from twisted.trial import unittest
def datagramReceived(self, data: bytes, partial: int, source: str, dest: str, protocol: int, version: int, ihl: int, tos: int, tot_len: int, fragment_id: int, fragment_offset: int, dont_fragment: int, more_fragments: int, ttl: int) -> None:
    assert self.expecting, 'Got a packet when not expecting anymore.'
    expectData, expectKw = self.expecting.pop(0)
    expectKwKeys = list(sorted(expectKw.keys()))
    localVariables = locals()
    for k in expectKwKeys:
        assert expectKw[k] == localVariables[k], f'Expected {k}={expectKw[k]!r}, got {localVariables[k]!r}'
    assert expectData == data, f'Expected {expectData!r}, got {data!r}'