import gyp.input
import unittest
def test_big_cycle(self):
    self._create_dependency(self.nodes['a'], self.nodes['b'])
    self._create_dependency(self.nodes['b'], self.nodes['c'])
    self._create_dependency(self.nodes['c'], self.nodes['d'])
    self._create_dependency(self.nodes['d'], self.nodes['e'])
    self._create_dependency(self.nodes['e'], self.nodes['a'])
    self.assertEqual([[self.nodes['a'], self.nodes['b'], self.nodes['c'], self.nodes['d'], self.nodes['e'], self.nodes['a']]], self.nodes['a'].FindCycles())