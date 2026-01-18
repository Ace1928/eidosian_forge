import unittest
from apitools.base.protorpclite import messages
from apitools.base.py import encoding
from apitools.base.py import exceptions
from apitools.base.py import util
def testTypecheck(self):

    class Class1(object):
        pass

    class Class2(object):
        pass

    class Class3(object):
        pass
    instance_of_class1 = Class1()
    self.assertEquals(instance_of_class1, util.Typecheck(instance_of_class1, Class1))
    self.assertEquals(instance_of_class1, util.Typecheck(instance_of_class1, ((Class1, Class2), Class3)))
    self.assertEquals(instance_of_class1, util.Typecheck(instance_of_class1, (Class1, (Class2, Class3))))
    self.assertEquals(instance_of_class1, util.Typecheck(instance_of_class1, Class1, 'message'))
    self.assertEquals(instance_of_class1, util.Typecheck(instance_of_class1, ((Class1, Class2), Class3), 'message'))
    self.assertEquals(instance_of_class1, util.Typecheck(instance_of_class1, (Class1, (Class2, Class3)), 'message'))
    with self.assertRaises(exceptions.TypecheckError):
        util.Typecheck(instance_of_class1, Class2)
    with self.assertRaises(exceptions.TypecheckError):
        util.Typecheck(instance_of_class1, (Class2, Class3))
    with self.assertRaises(exceptions.TypecheckError):
        util.Typecheck(instance_of_class1, Class2, 'message')
    with self.assertRaises(exceptions.TypecheckError):
        util.Typecheck(instance_of_class1, (Class2, Class3), 'message')