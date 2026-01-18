from ironicclient.tests.unit import utils
def check_exported_symbols(self, exported_symbols):
    self.assertIn('client', exported_symbols)
    self.assertIn('exc', exported_symbols)
    self.assertIn('exceptions', exported_symbols)