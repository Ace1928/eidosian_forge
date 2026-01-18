from ironicclient.tests.unit import utils
def test_default_import(self):
    default_imports = __import__(module_str, globals(), locals(), ['*'])
    exported_symbols = dir(default_imports)
    self.check_exported_symbols(exported_symbols)