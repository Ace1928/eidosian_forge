import sys
from unittest import TestCase
import simplejson as json
def test_failures(self):
    for idx, doc in enumerate(JSONDOCS):
        idx = idx + 1
        if idx in SKIPS:
            json.loads(doc)
            continue
        try:
            json.loads(doc)
        except json.JSONDecodeError:
            pass
        else:
            self.fail('Expected failure for fail%d.json: %r' % (idx, doc))