from unittest import TestCase
import simplejson as json
def test_listrecursion(self):
    x = []
    x.append(x)
    try:
        json.dumps(x)
    except ValueError:
        pass
    else:
        self.fail("didn't raise ValueError on list recursion")
    x = []
    y = [x]
    x.append(y)
    try:
        json.dumps(x)
    except ValueError:
        pass
    else:
        self.fail("didn't raise ValueError on alternating list recursion")
    y = []
    x = [y, y]
    json.dumps(x)