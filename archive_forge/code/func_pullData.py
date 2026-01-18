import sys
from qt import *
def pullData(self, fmt='chemical/daylight-smiles'):
    data = self.cdx.GetData(fmt)
    return str(data)