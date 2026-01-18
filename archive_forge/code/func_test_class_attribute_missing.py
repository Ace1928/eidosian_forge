import unittest
from traits.api import HasTraits, Instance, Str, Any, Property
def test_class_attribute_missing(self):
    """ This test demonstrates a problem with Traits objects with class
        attributes.  A change to the value of a class attribute via one
        instance causes the attribute to be removed from other instances.

        AttributeError: 'ClassWithClassAttribute' object has no attribute
        'name'
        """
    s = 'class defined name'
    c = ClassWithClassAttribute()
    self.assertEqual(s, c.name)
    c2 = ClassWithClassAttribute()
    self.assertEqual(s, c.name)
    self.assertEqual(s, c2.name)
    s2 = 'name class attribute changed via clone'
    c2.name = s2
    self.assertEqual(s2, c2.name)
    self.assertEqual(s, c.name)