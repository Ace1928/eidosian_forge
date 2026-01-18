from prov.model import *
def test_end_7(self):
    document = self.new_document()
    document.end(EX_NS['a1'], identifier=EX_NS['end7'], ender=EX_NS['a2'], time=datetime.datetime.now())
    self.do_tests(document)