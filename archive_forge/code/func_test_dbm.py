import shutil
from base64 import b64decode
from twisted.persisted import dirdbm
from twisted.python import rebuild
from twisted.python.filepath import FilePath
from twisted.trial import unittest
def test_dbm(self) -> None:
    d = self.dbm
    keys = []
    values = set()
    for k, v in self.items:
        d[k] = v
        keys.append(k)
        values.add(v)
    keys.sort()
    for k, v in self.items:
        self.assertIn(k, d)
        self.assertEqual(d[k], v)
    try:
        d[b'XXX']
    except KeyError:
        pass
    else:
        assert 0, "didn't raise KeyError on non-existent key"
    dbkeys = d.keys()
    dbvalues = set(d.values())
    dbitems = set(d.items())
    dbkeys.sort()
    items = set(self.items)
    self.assertEqual(keys, dbkeys, f".keys() output didn't match: {repr(keys)} != {repr(dbkeys)}")
    self.assertEqual(values, dbvalues, ".values() output didn't match: {} != {}".format(repr(values), repr(dbvalues)))
    self.assertEqual(items, dbitems, f"items() didn't match: {repr(items)} != {repr(dbitems)}")
    copyPath = self.mktemp()
    d2 = d.copyTo(copyPath)
    copykeys = d.keys()
    copyvalues = set(d.values())
    copyitems = set(d.items())
    copykeys.sort()
    self.assertEqual(dbkeys, copykeys, ".copyTo().keys() didn't match: {} != {}".format(repr(dbkeys), repr(copykeys)))
    self.assertEqual(dbvalues, copyvalues, ".copyTo().values() didn't match: %s != %s" % (repr(dbvalues), repr(copyvalues)))
    self.assertEqual(dbitems, copyitems, ".copyTo().items() didn't match: %s != %s" % (repr(dbkeys), repr(copyitems)))
    d2.clear()
    self.assertTrue(len(d2.keys()) == len(d2.values()) == len(d2.items()) == len(d2) == 0, '.clear() failed')
    self.assertNotEqual(len(d), len(d2))
    shutil.rmtree(copyPath)
    for k, v in self.items:
        del d[k]
        self.assertNotIn(k, d, 'key is still in database, even though we deleted it')
    self.assertEqual(len(d.keys()), 0, 'database has keys')
    self.assertEqual(len(d.values()), 0, 'database has values')
    self.assertEqual(len(d.items()), 0, 'database has items')
    self.assertEqual(len(d), 0, 'database has items')