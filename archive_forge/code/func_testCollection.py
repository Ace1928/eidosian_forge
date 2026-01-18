from twisted.python import roots
from twisted.trial import unittest
def testCollection(self) -> None:
    collection = roots.Collection()
    collection.putEntity('x', 'test')
    self.assertEqual(collection.getStaticEntity('x'), 'test')
    collection.delEntity('x')
    self.assertEqual(collection.getStaticEntity('x'), None)
    try:
        collection.storeEntity('x', None)
    except NotImplementedError:
        pass
    else:
        self.fail()
    try:
        collection.removeEntity('x', None)
    except NotImplementedError:
        pass
    else:
        self.fail()