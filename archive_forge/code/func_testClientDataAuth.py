import json
import sys
from pyu2f import errors
from pyu2f import model
def testClientDataAuth(self):
    cd = model.ClientData(model.ClientData.TYP_AUTHENTICATION, b'ABCD', 'somemachine')
    obj = json.loads(cd.GetJson())
    self.assertEquals(len(list(obj.keys())), 3)
    self.assertEquals(obj['typ'], model.ClientData.TYP_AUTHENTICATION)
    self.assertEquals(obj['challenge'], 'QUJDRA')
    self.assertEquals(obj['origin'], 'somemachine')