import math
import unittest
import os
from asyncio import iscoroutinefunction
from unittest.mock import AsyncMock, Mock, MagicMock, _magics
def test_asyncmock_defaults(self):
    mock = AsyncMock()
    self.assertEqual(int(mock), 1)
    self.assertEqual(complex(mock), 1j)
    self.assertEqual(float(mock), 1.0)
    self.assertNotIn(object(), mock)
    self.assertEqual(len(mock), 0)
    self.assertEqual(list(mock), [])
    self.assertEqual(hash(mock), object.__hash__(mock))
    self.assertEqual(str(mock), object.__str__(mock))
    self.assertTrue(bool(mock))
    self.assertEqual(round(mock), mock.__round__())
    self.assertEqual(math.trunc(mock), mock.__trunc__())
    self.assertEqual(math.floor(mock), mock.__floor__())
    self.assertEqual(math.ceil(mock), mock.__ceil__())
    self.assertTrue(iscoroutinefunction(mock.__aexit__))
    self.assertTrue(iscoroutinefunction(mock.__aenter__))
    self.assertIsInstance(mock.__aenter__, AsyncMock)
    self.assertIsInstance(mock.__aexit__, AsyncMock)
    self.assertEqual(oct(mock), '0o1')
    self.assertEqual(hex(mock), '0x1')