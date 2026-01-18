from ironicclient.tests.unit import utils
def test_import_objects(self):
    module = __import__(module_str)
    exported_symbols = dir(module)
    self.check_exported_symbols(exported_symbols)