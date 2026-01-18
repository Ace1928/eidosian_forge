import gyp.input
import unittest
def test_no_cycle_dag(self):
    self._create_dependency(self.nodes['a'], self.nodes['b'])
    self._create_dependency(self.nodes['a'], self.nodes['c'])
    self._create_dependency(self.nodes['b'], self.nodes['c'])
    for label, node in self.nodes.items():
        self.assertEqual([], node.FindCycles())