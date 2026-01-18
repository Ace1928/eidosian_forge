import random
import email.message
import pyzor
def init_for_sending(self):
    if 'Thread' not in self:
        self.set_thread(ThreadId.generate())
    assert 'Thread' in self
    self['PV'] = str(pyzor.proto_version)
    Message.init_for_sending(self)