from .common import TestCase
def test_group_completions(self):
    g = self.f.create_group('g')
    self.f.create_group('h')
    self.f.create_dataset('data', [1, 2, 3])
    self.assertEqual(self.f._ipython_key_completions_(), ['data', 'g', 'h'])
    self.f.create_group('data2', [1, 2, 3])
    self.assertEqual(self.f._ipython_key_completions_(), ['data', 'data2', 'g', 'h'])
    g.create_dataset('g_data1', [1, 2, 3])
    g.create_dataset('g_data2', [4, 5, 6])
    self.assertEqual(g._ipython_key_completions_(), ['g_data1', 'g_data2'])
    g.create_dataset('g_data3', [7, 8, 9])
    self.assertEqual(g._ipython_key_completions_(), ['g_data1', 'g_data2', 'g_data3'])