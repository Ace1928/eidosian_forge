from prov.model import *
def test_activity_3(self):
    document = self.new_document()
    document.activity(EX_NS['a3'], startTime=datetime.datetime.now(), endTime=datetime.datetime.now())
    self.do_tests(document)