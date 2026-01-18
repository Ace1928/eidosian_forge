import asyncio
import contextvars
import unittest
from test import support
def on_cleanup1(self):
    self.assertEqual(events, expected[:10])
    events.append('cleanup1')
    VAR.set(VAR.get() + ('cleanup1',))
    nonlocal cvar
    cvar = VAR.get()