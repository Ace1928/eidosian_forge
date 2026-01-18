from prov.model import *
def test_mention_2(self):
    document = self.new_document()
    document.mention(EX_NS['e2'], EX_NS['e1'], EX_NS['b'])
    self.do_tests(document)